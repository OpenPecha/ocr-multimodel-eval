import os
import json
import logging
import csv
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.cloud import vision
from google.cloud.vision import AnnotateImageResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ocr_multimodel_eval.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Initialize Google Vision client
try:
    # Set the path to the service account key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../google_vision_key.json'
    vision_client = vision.ImageAnnotatorClient()
    logging.info("Google Vision client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Google Vision client: {e}")
    vision_client = None

# Initialize Gemini client
gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_client = None
if gemini_api_key:
    try:
        gemini_client = genai.Client(api_key=gemini_api_key)
        logging.info("Gemini client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        gemini_client = None
else:
    logging.warning("GEMINI_API_KEY not found in environment variables. Gemini models will not be available.")

# Supported models
SUPPORTED_MODELS = {
    'google_vision': 'Google Cloud Vision API',
    'gemini_2_5_pro': 'gemini-2.5-pro',
    'gemini_2_5_flash': 'gemini-2.5-flash',
    'gemini_2_5_flash_lite': 'gemini-2.5-flash-lite',
    'gemini_2_0_flash': 'gemini-2.0-flash',
    'gemini_2_0_flash_lite': 'gemini-2.0-flash-lite'
}

class OCRMultiModelEvaluator:
    """Multi-model OCR evaluator supporting Google Vision and Gemini models."""
    
    def __init__(self):
        self.vision_client = vision_client
        self.gemini_client = gemini_client
        

    
    def google_vision_ocr(self, image_path: str, lang_hint: Optional[str] = None) -> Dict[str, Any]:
        """Perform OCR using Google Cloud Vision API."""
        try:
            image = vision.Image()
            
            # Load local image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image.content = content
            
            features = [
                {
                    "type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
                    "model": "builtin/weekly",
                }
            ]
            
            image_context = {}
            if lang_hint:
                image_context["language_hints"] = [lang_hint]
            
            response = self.vision_client.annotate_image({
                "image": image,
                "features": features,
                "image_context": image_context
            })
            
            response_json = AnnotateImageResponse.to_json(response)
            response_dict = json.loads(response_json)
            
            # Extract text from response
            if "error" in response_dict:
                text = ""
                error = response_dict["error"]
            elif "fullTextAnnotation" in response_dict:
                text = response_dict["fullTextAnnotation"].get("text", "")
                error = None
            elif "textAnnotations" in response_dict and response_dict["textAnnotations"]:
                text = response_dict["textAnnotations"][0].get("description", "")
                error = None
            else:
                text = ""
                error = None
            
            return {
                "model": "google_vision",
                "text": text.strip(),
                "raw_response": response_dict,
                "error": error
            }
            
        except Exception as e:
            logging.error(f"Google Vision OCR error: {e}")
            return {
                "model": "google_vision",
                "text": "",
                "raw_response": None,
                "error": str(e)
            }
    
    def gemini_ocr(self, image_path: str, model_name: str, lang_hint: Optional[str] = None) -> Dict[str, Any]:
        """Perform OCR using Gemini models with retry logic."""
        try:
            if not self.gemini_client:
                raise ValueError("Gemini client not initialized. Check GEMINI_API_KEY in environment variables")
            
            # Get the actual model name
            if model_name not in SUPPORTED_MODELS:
                raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(SUPPORTED_MODELS.keys())}")
            
            actual_model_name = SUPPORTED_MODELS[model_name]
            
            # Load image as raw bytes to preserve quality
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Determine MIME type from file extension
            image_path_lower = image_path.lower()
            if image_path_lower.endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif image_path_lower.endswith('.png'):
                mime_type = 'image/png'
            elif image_path_lower.endswith('.webp'):
                mime_type = 'image/webp'
            else:
                # Default to JPEG if unknown
                mime_type = 'image/jpeg'
            
            # Base prompt
            prompt = """ Please OCR and extract all the main text from this file, be as accurate as possible. Make use of your specific knowledge of Tibetan to ensure accuracy. Don't use markdown in the output. """
            
            if lang_hint:
                prompt += f" The text is in {lang_hint} language."
            
            # Use the engineer's OCR approach with retry logic
            text_result = self._ocr_image_with_retry(image_bytes, mime_type, prompt, actual_model_name)
            
            return {
                "model": model_name,
                "text": text_result if text_result != "ERROR" else "",
                "raw_response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": text_result
                            }]
                        }
                    }] if text_result != "ERROR" else []
                },
                "error": "OCR processing failed after retries" if text_result == "ERROR" else None
            }
            
        except Exception as e:
            logging.error(f"Gemini OCR error with {model_name}: {e}")
            return {
                "model": model_name,
                "text": "",
                "raw_response": None,
                "error": str(e)
            }
    
    def _ocr_image_with_retry(self, image_bytes: bytes, mime_type: str, prompt: str, model_name: str) -> str:
        """Process a single image using Gemini model with retry logic (based on engineer's code)."""
        import time
        from google.genai.types import GenerateContentConfig, ThinkingConfig, Part
        
        temp = 1
        for attempt in range(5):
            try:
                # Set thinking budget based on model requirements
                thinking_budget = 128 if 'pro' in model_name.lower() else 0
                
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                    config=GenerateContentConfig(
                        system_instruction="You are an absolute expert on Tibetan.",
                        temperature=temp,
                        max_output_tokens=4000,
                        thinking_config=ThinkingConfig(thinking_budget=thinking_budget)
                    )
                )
                print("Got Response")
                if response.text is not None:
                    return response.text.strip()
                else:
                    print(f"[OCR_IMAGE] Attempt {attempt+1}, response.text is None")
                    temp += 0.2
                    time.sleep(3)
                    continue
                    
            except Exception as e:
                temp += 0.2
                print(f"[OCR_IMAGE] Attempt {attempt+1}, error: {e}")
                time.sleep(3)
                continue
                
        return "ERROR"
    
    def run_ocr(self, image_path: str, model_name: str, lang_hint: Optional[str] = None, save_to_csv: bool = True) -> Dict[str, Any]:
        """Run OCR using the specified model.
        
        Args:
            image_path: Path to local image file
            model_name: Name of the model to use (see SUPPORTED_MODELS)
            lang_hint: Optional language hint (e.g., 'bo' for Tibetan)
            save_to_csv: Whether to save results to CSV file (default: True)
            
        Returns:
            Dictionary containing OCR results
        """
        logging.info(f"Running OCR with model: {model_name} on image: {image_path}")
        
        if model_name not in SUPPORTED_MODELS:
            result = {
                "model": model_name,
                "text": "",
                "raw_response": None,
                "error": f"Unsupported model: {model_name}. Supported models: {list(SUPPORTED_MODELS.keys())}"
            }
        elif model_name == 'google_vision':
            result = self.google_vision_ocr(image_path, lang_hint)
        else:
            result = self.gemini_ocr(image_path, model_name, lang_hint)
        
        # Save to CSV if requested
        if save_to_csv:
            self.save_result_to_csv(image_path, result, model_name)
        
        return result
    
    def save_result_to_csv(self, image_path: str, result: Dict[str, Any], model_name: str):
        """Save OCR result to CSV file.
        
        Args:
            image_path: Path to image file
            result: OCR result dictionary
            model_name: Name of the model used
        """
        try:
            # Extract filename from path
            filename = os.path.basename(image_path)
            image_name_without_ext = os.path.splitext(filename)[0]
            
            # Create model-specific output directory
            output_dir = os.path.join("../data/script_inferenced_2", model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create CSV filename with just the image filename (no model prefix since it's in model folder)
            csv_filename = f"{image_name_without_ext}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Prepare data for CSV with proper Tibetan text handling
            inference_text = result.get('text', '')
            error_text = result.get('error', '')
            
            # Clean and normalize Tibetan text
            if inference_text:
                # Remove any extra whitespace and normalize
                inference_text = ' '.join(inference_text.split())
            
            csv_data = {
                'image_filename': filename,
                'inference_result': inference_text,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat(),
                'error': error_text
            }
            
            # Check if CSV file exists
            file_exists = os.path.isfile(csv_path)
            
            # Write to CSV with UTF-8 BOM for better Excel compatibility
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['image_filename', 'inference_result', 'model_used', 'timestamp', 'error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                    logging.info(f"Created new CSV file with UTF-8 BOM: {csv_path}")
                
                writer.writerow(csv_data)
                logging.info(f"Saved result to CSV: {csv_path}")
                
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
    
    def process_batch_images(self, image_list: list, model_name: str, lang_hint: Optional[str] = None) -> list:
        """Process a batch of images with the specified model.
        
        Args:
            image_list: List of image paths
            model_name: Name of the model to use
            lang_hint: Optional language hint
            
        Returns:
            List of OCR results
        """
        results = []
        total_images = len(image_list)
        
        logging.info(f"Processing {total_images} images with model: {model_name}")
        
        for i, image_path in enumerate(image_list, 1):
            logging.info(f"Processing image {i}/{total_images}: {image_path}")
            
            result = self.run_ocr(image_path, model_name, lang_hint, save_to_csv=True)
            results.append(result)
            
            # Log progress
            if i % 10 == 0 or i == total_images:
                logging.info(f"Completed {i}/{total_images} images")
        
        logging.info(f"Batch processing completed for model: {model_name}")
        return results
    
    def combine_csv_files(self, model_name: str) -> str:
        """Combine all individual CSV files for a model into one consolidated CSV file.
        
        Args:
            model_name: Name of the model to combine CSV files for
            
        Returns:
            Path to the combined CSV file
        """
        try:
            # Define paths
            model_dir = os.path.join("../data/script_inferenced_2", model_name)
            combined_csv_path = os.path.join(model_dir, f"{model_name}_combined.csv")
            
            if not os.path.exists(model_dir):
                logging.warning(f"Model directory not found: {model_dir}")
                return ""
            
            # Get all CSV files in the model directory (excluding any existing combined file)
            csv_files = []
            for filename in os.listdir(model_dir):
                if filename.endswith('.csv') and not filename.endswith('_combined.csv'):
                    csv_files.append(os.path.join(model_dir, filename))
            
            if not csv_files:
                logging.warning(f"No individual CSV files found in {model_dir}")
                return ""
            
            # Sort files naturally (img_1.csv, img_2.csv, etc.)
            import re
            def natural_sort_key(path):
                filename = os.path.basename(path)
                numbers = re.findall(r'\d+', filename)
                return [int(num) if num.isdigit() else num for num in numbers] if numbers else [filename]
            
            csv_files.sort(key=natural_sort_key)
            
            # Combine all CSV files
            combined_data = []
            fieldnames = ['image_filename', 'inference_result', 'model_used', 'timestamp', 'error']
            
            for csv_file in csv_files:
                try:
                    with open(csv_file, 'r', encoding='utf-8-sig') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            combined_data.append(row)
                except Exception as e:
                    logging.error(f"Error reading {csv_file}: {e}")
                    continue
            
            # Write combined CSV file
            with open(combined_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(combined_data)
            
            logging.info(f"Combined {len(csv_files)} CSV files into: {combined_csv_path}")
            logging.info(f"Total rows in combined file: {len(combined_data)}")
            
            return combined_csv_path
            
        except Exception as e:
            logging.error(f"Error combining CSV files: {e}")
            return ""
    
    def process_multimodel_images(self, model_name: str, lang_hint: Optional[str] = None, images_folder: str = "../data/images") -> list:
        """Process all images in the multimodel-images folder with the specified model.
        
        Args:
            model_name: Name of the model to use
            lang_hint: Optional language hint (e.g., 'bo' for Tibetan)
            images_folder: Folder containing images (default: 'multimodel-images')
            
        Returns:
            List of OCR results
        """
        try:
            # Get list of image files from the folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = []
            
            for filename in os.listdir(images_folder):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(images_folder, filename)
                    image_files.append(image_path)
            
            # Sort files to ensure proper order (img_1, img_2, img_3, etc.)
            import re
            def natural_sort_key(path):
                filename = os.path.basename(path)
                # Extract numbers from filename for proper sorting
                numbers = re.findall(r'\d+', filename)
                return [int(num) if num.isdigit() else num for num in numbers] if numbers else [filename]
            
            image_files.sort(key=natural_sort_key)
            
            if not image_files:
                logging.warning(f"No image files found in {images_folder}")
                return []
            
            logging.info(f"Found {len(image_files)} images in {images_folder}")
            logging.info(f"Processing with model: {model_name}")
            
            # Process all images
            return self.process_batch_images(image_files, model_name, lang_hint)
            
        except FileNotFoundError:
            logging.error(f"Folder {images_folder} not found")
            return []
        except Exception as e:
            logging.error(f"Error processing multimodel images: {e}")
            return []


def main():
    """Example usage of the OCR Multi-Model Evaluator."""
    evaluator = OCRMultiModelEvaluator()
    
    # Example usage
    print("OCR Multi-Model Evaluator")
    print("Supported models:", list(SUPPORTED_MODELS.keys()))
    
    # -----Example: Run OCR on a single image with a specific model-------

    # result = evaluator.run_ocr("../data/images/img_7.jpg", "gemini_2_0_flash_lite", lang_hint=None, save_to_csv=True)
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    # -----------------------------------------------------------------

    # ***********Get all Gemini models (exclude models of choice)***********

    gemini_models = [model for model in SUPPORTED_MODELS.keys() if model != "google_vision"]
    print(f"\nProcessing with Gemini models: {gemini_models}")
    
    # Process all images with each Gemini model
    for i, model_name in enumerate(gemini_models, 1):
        print(f"\n{'='*60}")
        print(f"Processing with model {i}/{len(gemini_models)}: {model_name}")
        print(f"{'='*60}")
        
        # Process all images in the folder
        results = evaluator.process_multimodel_images(model_name, lang_hint=None, images_folder="../data/images")
        print(f"Processed {len(results)} images with {model_name}")
        
        # Combine all individual CSV files into one consolidated file
        combined_csv_path = evaluator.combine_csv_files(model_name)
        if combined_csv_path:
            print(f"Combined CSV file created: {combined_csv_path}")
        else:
            print(f"Failed to create combined CSV file for {model_name}")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE! Processed all {len(gemini_models)} Gemini models.")
    print(f"Results saved in: data/script_inferenced_2/")
    print(f"{'='*60}")
    
    # *************************************************************

    # -------------- Example: Process specific images --------------
    # image_list = ["../data/images/img_1.jpg", "../data/images/img_2.png", "../data/images/img_3.png", "../data/images/img_4.jpg", "../data/images/img_5.png", "../data/images/img_6.jpg", "../data/images/img_7.jpg"]
    # results = evaluator.process_batch_images(image_list, "google_vision", lang_hint=None)
    # print(f"Processed {len(results)} specific images")
    # evaluator.combine_csv_files("google_vision")
    # -----------------------------------------------------------------


if __name__ == "__main__":
    main()