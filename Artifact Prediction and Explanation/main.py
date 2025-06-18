import os
import argparse
import pandas as pd
from tqdm import tqdm
import json
import re
from utils import *
from cifar_inference import load_dinov2, load_svm, run_cifar_inference
from cnn_inference import load_densenet, load_resnet, run_cnn_inference
from clip_inference import run_clip
from convert_to_json import parse_artifacts, json_formatting
from constants import artifact_index_dict, cifar_class_dict
from combine_algorithm import run_combination_algorithm
from artifact_explanation import artifact_explainer
from ovis_inference import load_ovis


def main(folder_path, tsv_file_path, svm_path, cnn_models_base_path, num_preds, limit_flag, clip_list_limit):
    """
    Runs the full pipeline for artifact prediction and explanation on a set of images.

    The pipeline involves:
    - Class prediction using the DinoV2 model.
    - Artifact prediction using CNN models (ResNet and DenseNet).
    - Artifact prediction using the Jina-Clip-v2 model.
    - Optimally combining artifact predictions using the combination algorithm.
    - Generating artifact explanations using the Ovis model.

    Args:
        folder_path (str): The directory containing the images to be processed.
        tsv_file_path (str): Path to the TSV file containing data for class labels and artifact mappings.
        svm_path (str): Path to the SVM model used for CIFAR classification.
        cnn_models_base_path (str): Base directory for loading CNN models (ResNet and DenseNet).
        num_preds (int): The number of artifacts to output per image.
        limit_flag (int): A flag to determine if similar classes should be counted towards the prediction limit.
        clip_list_limit (int): Limits the size of the list of predictions from the Clip model.

    Returns:
        None: The function writes the final output as a JSON file (`output.json`).
    """
    # Loads all the models required for artifact prediction and explanation
    dataframe = pd.read_csv(tsv_file_path, sep="\t")
    dinov2_model = load_dinov2()
    svm_model = load_svm(svm_path)
    resnet_dict = load_resnet(cnn_models_base_path)
    densenet_dict = load_densenet(cnn_models_base_path)
    ovis, text_tokenizer, visual_tokenizer = load_ovis()

    # Initialize the JSON output list
    json_output = []

    # Runs the pipeline for each image
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        
        # Skip the current iteration if the file is not a PNG image
        if not filename.lower().endswith('.png'):
            continue

        # Class prediction by DinoV2
        object_class = run_cifar_inference(img_path, dinov2_model, svm_model)

        # Artifact prediction by CNN models
        cnn_artifacts_dict = cnn_parser(dataframe["CNN"][object_class])
        cnn_models_dict = resnet_dict | densenet_dict
        cnn_output_dict = run_cnn_inference(img_path, cnn_models_dict, cnn_artifacts_dict)

        # Artifact prediction by Jina-Clip-v2
        clip_prediction = run_clip(img_path, object_class, dataframe, clip_list_limit, 0)

        # Prepares the final prediction list using combination algorithm
        prediction_list = run_combination_algorithm(dataframe, object_class, clip_prediction, cnn_output_dict, num_preds, limit_flag)
        num_of_artifacts = len(prediction_list)
        predicted_artifact_list = [key for key, value in artifact_index_dict.items() if value in prediction_list]

        # Runs Ovis1.6-Gemma2-9B to generate explanation for each predicted artifact
        ovis_output = artifact_explainer(img_path, cifar_class_dict[object_class], predicted_artifact_list, num_of_artifacts, ovis, text_tokenizer, visual_tokenizer)

        # Convert the ovis output to the requisite JSON format
        parsed_data = parse_artifacts(ovis_output)
        json_output = json_formatting(filename, parsed_data, json_output)
        with open("output.json", "w") as json_file:
            json.dump(json_output, json_file, indent=4)



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run image inference and compute IOU.")

    parser.add_argument('--folder_path', type=str, help="Path to the image folder.")
    parser.add_argument('--tsv_file_path', type=str, help="Path to the TSV file.")
    parser.add_argument('--svm_path', type=str, help="Path to the SVM used to classify CIFAR classes.")
    parser.add_argument('--cnn_model_base_path', type=str, help="Base path for CNN models.")
    parser.add_argument('--num_preds', type=int, default=12, help="Number of predictions to make.")
    parser.add_argument('--limit_flag', type=int, default=0, help="Determines if similar classes account towards the prediction limit.")
    parser.add_argument('--clip_list_limit', type=int, default = 15, help="Limits the size of the clip prediction list")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.folder_path, args.tsv_file_path, args.svm_path, args.cnn_model_base_path, args.num_preds, args.limit_flag, args.clip_list_limit)



