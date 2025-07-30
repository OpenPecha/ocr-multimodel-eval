import os
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pyewts
from evaluate import load


def calculate_cer(reference: str, hypothesis: str, converter, cer_scorer) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis texts
    using pyewts for Wylie transliteration and evaluate library's CER scorer.
    
    Args:
        reference (str): Ground truth text
        hypothesis (str): Predicted text
        converter: pyewts converter instance
        cer_scorer: evaluate CER scorer instance
        
    Returns:
        float: CER value as a decimal (0.0 to 1.0)
    """
    # Handle empty cases
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    if not hypothesis.strip():
        return 1.0
    
    try:
        # Convert both texts to Wylie transliteration
        reference_wylie = converter.toWylie(reference.strip())
        hypothesis_wylie = converter.toWylie(hypothesis.strip())
        
        # Calculate CER using evaluate library
        cer_score = cer_scorer.compute(predictions=[hypothesis_wylie], references=[reference_wylie])
        return cer_score
    except Exception as e:
        print(f"[ERROR] CER calculation failed: {e}")
        return 1.0  # Return maximum error on failure


def parse_ground_truth(file_path: str) -> Dict[int, str]:
    """
    Parse ground truth file and extract text for each image.
    
    Args:
        file_path (str): Path to ground truth file
        
    Returns:
        Dict[int, str]: Dictionary mapping image number to text content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by "Image X" pattern
    parts = re.split(r'Image (\d+)', content)
    
    # Process parts (skip first empty part if exists)
    ground_truth = {}
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            image_num = int(parts[i])
            image_text = parts[i + 1].strip()
            ground_truth[image_num] = image_text
    
    return ground_truth


def parse_inference_file(file_path: str) -> Dict[int, str]:
    """
    Parse inference file and extract text for each image.
    
    Args:
        file_path (str): Path to inference file
        
    Returns:
        Dict[int, str]: Dictionary mapping image number to text content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by "Image X" pattern
    parts = re.split(r'Image (\d+)', content)
    
    # Process parts (skip first empty part if exists)
    inference_texts = {}
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            image_num = int(parts[i])
            image_text = parts[i + 1].strip()
            inference_texts[image_num] = image_text
    
    return inference_texts


def calculate_model_cer(ground_truth: Dict[int, str], inference: Dict[int, str], converter, cer_scorer) -> Tuple[List[float], float]:
    """
    Calculate CER for each image and combined CER for a model.
    
    Args:
        ground_truth (Dict[int, str]): Ground truth texts by image
        inference (Dict[int, str]): Inference texts by image
        converter: pyewts converter instance
        cer_scorer: evaluate CER scorer instance
        
    Returns:
        Tuple[List[float], float]: Individual CER values (as percentages) and combined CER
    """
    individual_cers = []
    all_references = []
    all_predictions = []
    
    # Calculate CER for each image
    for img_num in sorted(ground_truth.keys()):
        if img_num in inference:
            gt_text = ground_truth[img_num]
            inf_text = inference[img_num]
            
            # Calculate individual CER (returns decimal 0.0-1.0)
            cer_value = calculate_cer(gt_text, inf_text, converter, cer_scorer)
            individual_cers.append(round(cer_value, 4))
            
            # Collect for combined CER calculation
            if gt_text.strip() and inf_text.strip():
                all_references.append(gt_text.strip())
                all_predictions.append(inf_text.strip())
        else:
            # If image not found in inference, treat as maximum error (1.0)
            individual_cers.append(1.0)
            if ground_truth[img_num].strip():
                all_references.append(ground_truth[img_num].strip())
                all_predictions.append("")  # Empty prediction
    
    # Calculate combined CER using all texts together
    if all_references and all_predictions:
        try:
            # Convert all references and predictions to Wylie
            ref_wylie = [converter.toWylie(ref) for ref in all_references]
            pred_wylie = [converter.toWylie(pred) for pred in all_predictions]
            
            # Calculate combined CER
            combined_cer = round(cer_scorer.compute(predictions=pred_wylie, references=ref_wylie), 4)
        except Exception as e:
            print(f"[ERROR] Combined CER calculation failed: {e}")
            # Fallback to average of individual CERs
            combined_cer = round(sum(individual_cers) / len(individual_cers), 4) if individual_cers else 1.0
    else:
        combined_cer = 1.0
    
    return individual_cers, combined_cer


def save_results_to_csv(model_name: str, individual_cers: List[float], combined_cer: float, ground_truth_texts: Dict[int, str], inference_texts: Dict[int, str], output_dir: str):
    """
    Save CER results to CSV file with ground truth and inference texts.
    
    Args:
        model_name (str): Name of the model
        individual_cers (List[float]): CER values for each image
        combined_cer (float): Combined CER value
        ground_truth_texts (Dict[int, str]): Ground truth texts by image number
        inference_texts (Dict[int, str]): Inference texts by image number
        output_dir (str): Output directory path
    """
    csv_path = os.path.join(output_dir, f"{model_name}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Image', 'CER', 'Ground_Truth', 'Inferenced'])
        
        # Write individual image CERs with texts
        for i, cer in enumerate(individual_cers, 1):
            gt_text = ground_truth_texts.get(i, "")
            inf_text = inference_texts.get(i, "")
            
            # Clean texts for CSV (remove newlines and excessive whitespace)
            gt_clean = ' '.join(gt_text.split()) if gt_text else ""
            inf_clean = ' '.join(inf_text.split()) if inf_text else ""
            
            writer.writerow([f'Image {i}', cer, gt_clean, inf_clean])
        
        # Write combined CER (no text for combined row)
        writer.writerow(['Combined', combined_cer, '', ''])
    
    print(f"Results saved to: {csv_path}")


def main():
    """
    Main function to execute CER calculation for all models.
    """
    # paths
    project_root = Path(__file__).parent.parent
    ground_truth_path = project_root / "data" / "ground_truth" / "ground_truth_of_multi_models.txt"
    inference_dir = project_root / "data" / "inferenced"
    results_dir = project_root / "results"
    
    results_dir.mkdir(exist_ok=True)
    
    print("Starting CER Multi-Model Evaluation...")
    print(f"Ground truth file: ***{ground_truth_path}***")
    print(f"Inference directory: ***{inference_dir}***")
    print(f"Results directory: ***{results_dir}***")
    print("-" * 60)
    
    # Initialize pyewts converter and CER scorer
    print("\n****Initializing pyewts converter and CER scorer****...")
    try:
        converter = pyewts.pyewts()
        cer_scorer = load("cer")
        print("✅ Successfully initialized evaluation tools")
    except Exception as e:
        print(f"[ERROR] Failed to initialize evaluation tools: {e}")
        print("Please ensure pyewts and evaluate libraries are installed:")
        print("pip install pyewts evaluate")
        return
    
    # Parse ground truth
    print("\n****Parsing ground truth file****...")
    ground_truth = parse_ground_truth(str(ground_truth_path))
    print(f"Found ***{len(ground_truth)}*** images in ground truth")
    
    # Process each inference file
    inference_files = list(inference_dir.glob("*.txt"))
    print(f"Found ***{len(inference_files)}*** inference files")
    
    for inference_file in inference_files:
        model_name = inference_file.stem  # Get filename without extension
        print(f"\n****Processing {model_name}****...")
        
        # Parse inference file
        inference_data = parse_inference_file(str(inference_file))
        print(f"Found ***{len(inference_data)}*** images in {model_name}")
        
        # Calculate CER
        individual_cers, combined_cer = calculate_model_cer(ground_truth, inference_data, converter, cer_scorer)
        
        # Display results
        print(f"Individual CERs: ***{individual_cers}***")
        print(f"Combined CER: ***{combined_cer}***")
        
        # Save to CSV
        save_results_to_csv(model_name, individual_cers, combined_cer, ground_truth, inference_data, str(results_dir))
    
    print("\n" + "="*60)
    print("CER Multi-Model Evaluation completed successfully!")
    print(f"All results saved in: {results_dir}")


if __name__ == "__main__":
    main()