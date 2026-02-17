import os
import urllib.parse
import hashlib
import requests
from PIL import Image
import time
import chromadb

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.console import Console, Group
import itertools

import pandas as pd
from typing import Any, List, Dict
import lmdb
import json
from mwviews.api import PageviewsClient
from collections import defaultdict


def filter_entity_attributes(entity_info_dict) -> Dict[str, Any]:
    """Filter out unnecessary entity attributes."""
    filter_attribute_list = ["image_size", "website", "description", "coordinates"]
    entity_attributes = entity_info_dict.get("entity_attributes")
    if not entity_attributes or not isinstance(entity_attributes, dict):
        return entity_info_dict

    for key in list(entity_attributes.keys()):
        if key in filter_attribute_list:
            del entity_attributes[key]

    return entity_info_dict

def ensure_crag_cache_dir_is_configured():
    """
    Ensure the cache directory for CRAG images exists and is properly configured.
    
    This function:
    1. Checks if CRAG_CACHE_DIR environment variable is set
    2. If not set, uses platform-appropriate default cache location
    3. Creates the directory if it doesn't exist
    4. Returns the path to the cache directory
    
    Returns:
        str: Path to the cache directory
    """    
    # First check if user has explicitly set a cache directory
    cache_dir = os.environ.get("CRAG_CACHE_DIR")
    
    if not cache_dir:
        # Use platform-specific default locations if not explicitly set
        if os.name == 'nt':  # Windows
            cache_home = os.environ.get("LOCALAPPDATA", os.path.expanduser("~/AppData/Local"))
        else:  # Unix/Linux/Mac
            cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        
        cache_dir = os.path.join(cache_home, "cragmm_images_cache")
        
        # Print info message only the first time
        if not hasattr(ensure_crag_cache_dir_is_configured, "_cache_location_shown"):
            print(f"Caching downloaded images in {cache_dir}")
            print("You can override this by setting the CRAG_CACHE_DIR environment variable.")
            ensure_crag_cache_dir_is_configured._cache_location_shown = True
    
    # Create the directory if it doesn't exist
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create cache directory {cache_dir}: {e}")
            # Fall back to a temporary directory if we can't create the default
            import tempfile
            cache_dir = os.path.join(tempfile.gettempdir(), "cragmm_images_cache")
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using fallback cache directory: {cache_dir}")
    
    return cache_dir    

def download_image_url(image_url):
    """Downloads image from URL and saves it to the cache directory with a deterministic name.
    Returns local path if successful, raises Exception otherwise.
    
    Args:
        image_url: URL of the image to download
        
    Returns:
        str: Local path to the downloaded or cached image
        
    Raises:
        Exception: If the image couldn't be downloaded or is invalid
    """
    cache_dir = ensure_crag_cache_dir_is_configured()
    
    # Create cache directory if it doesn't exist (redundant but keeps backward compatibility)
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Create a deterministic filename based on the URL
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        file_extension = os.path.splitext(image_url.split('?')[0])[1] or '.jpg'
        local_filename = f"{url_hash}{file_extension}"
        local_path = os.path.join(cache_dir, local_filename)
        
        # If the file already exists in cache, validate and return it
        if os.path.exists(local_path):
            if _is_valid_image(local_path):
                print(f"Using cached image from {local_path}")
                return local_path
            else:
                print(f"Cached image is invalid, re-downloading: {local_path}")
                # Continue with download as the cached file is invalid
        
        # Download the image
        headers = {"User-Agent": "CRAGBot/v0.0.1"}
        response = requests.get(image_url, stream=True, timeout=5, headers=headers)
        response.raise_for_status()
        
        # Save the image to a temporary file first
        temp_path = f"{local_path}.temp"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Validate the downloaded image
        if _is_valid_image(temp_path):
            # Move to final location if valid
            os.replace(temp_path, local_path)
            print(f"Downloaded and validated image_url to {local_path}")
            return local_path
        else:
            # Remove invalid image
            os.remove(temp_path)
            print(f"Downloaded image is not valid from URL: {image_url}")
            raise Exception(f"Downloaded image is not valid from URL {image_url}")
            
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        raise Exception(f"Error downloading image from {image_url}: {e}")

def _is_valid_image(file_path):
    """Check if the file is a valid image using PIL.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            # Verify the image by loading it
            img.verify()
            
            # Additional check by accessing image properties
            width, height = img.size
            if width <= 0 or height <= 0:
                return False
                
            return True
    except Exception as e:
        print(f"Invalid image file {file_path}: {e}")
        return False

def is_url(url):
    """Check if the URL is a valid image URL."""
    try:
        result = urllib.parse.urlparse(url)
        return bool(result.scheme and result.netloc)
    except Exception:
        return False
    
def display_results(console: Console, turn_evaluation_results_df: pd.DataFrame, scores_dictionary: Dict[str, Any], display_conversations: int = 3, is_ambig: bool = False) -> None:
    """Display evaluation results in a formatted way"""
    
    title = "Evaluation Results" if not is_ambig else "Ambiguous Cases Evaluation Results"

    # Create metrics table
    metrics_table = Table(show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value")

    metrics_table.add_row("Total conversations", str(turn_evaluation_results_df["session_id"].nunique()))
    metrics_table.add_row("Exact matches", str(scores_dictionary["correct_exact"]))
    metrics_table.add_row("Semantic matches", str(scores_dictionary["correct_semantic"]))
    metrics_table.add_row("Exact accuracy", f"{scores_dictionary['exact_match']:.2%}")
    metrics_table.add_row("Accuracy", f"{scores_dictionary['accuracy']:.2%}")
    
    # Create a list of renderables to display in the panel
    renderables = [metrics_table]
    
    # Add sample conversation tables if requested
    if display_conversations > 0:
        def _init_conversation_table(title: str):
            table = Table(title=title)
            table.add_column("session_id", style="dim")
            table.add_column("Query", style="dim")
            table.add_column("Agent Response", style="dim")
            table.add_column("Ground Truth", style="dim")
            table.add_column("API Response", style="dim")
            table.add_column("Evaluation Result", style="dim")
            return table
        
        def _get_status_style_and_text(row: pd.Series):
            if row["is_exact_match"]:
                return "green", "[green]EXACT MATCH[/green]"
            elif row["is_semantically_correct"]:
                return "green", "[green]SEMANTICALLY CORRECT[/green]"
            else:
                return "red", "[red]INCORRECT[/red]"
        
        # Add section header for sample results
        renderables.append(Text("\nSample Evaluation Results", style="bold cyan"))
        
        table = _init_conversation_table("Evaluation Results")
        for idx, row in itertools.islice(turn_evaluation_results_df.iterrows(), display_conversations):
            status_style, status_text = _get_status_style_and_text(row)
            table.add_row(  
                        f"{row['session_id'][:5]}",
                        f"[bold cyan]{row['query']}[/bold cyan]", 
                        f"[bold yellow]{row['agent_response']}[/bold yellow]", 
                        f"[bold green]{row['ground_truth']}[/bold green]", 
                        f"[bold blue]{str(row['api_response'])[:100]}[/bold blue]", 
                        f"[bold {status_style}]{status_text}[/bold {status_style}]"
                        )
        renderables.append(table)

    # Display all tables in a single panel using Group to combine renderables
    group = Group(*renderables)
    panel = Panel(
        group,
        title=f"[bold]{title}[/bold]",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)

def maybe_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    else:
        return [x]

def hash_key(key: str) -> str:
    # returns a 64-char hex string
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def shard_id(hashed_key: str, num_shards: int) -> int:
    h = int(hashed_key, 16)
    return h % num_shards


def lookup_web_content(db_dir, key):
    num_shards = max([int(i.split('.mdb')[0].replace('shard_', '')) for i in os.listdir(db_dir) if '.mdb' in i]) + 1
    hashed_key = hash_key(key)
    sid = shard_id(hashed_key, num_shards)
    env = lmdb.open(
        os.path.join(db_dir, f"shard_{sid}.mdb"),
        readonly=True,
        lock=False,
        subdir=False
    )
    with env.begin() as txn:
        raw = txn.get(hashed_key.encode("utf-8"))
    env.close()
    return json.loads(raw) if raw else None

def convert_bbox_to_resized_format(bbox, orig_size, target_size=(960, 1280)):
    """
    Convert bbox from original image size to target size.
    bbox: [x1, y1, x2, y2] in original image coordinates
    orig_size: (width, height) of original image
    target_size: (width, height) of target image
    Returns: [x1, y1, x2, y2] in target image coordinates
    """
    orig_w, orig_h = orig_size
    target_w, target_h = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    x1, y1, x2, y2 = bbox
    x1_new = x1 * scale_x
    y1_new = y1 * scale_y
    x2_new = x2 * scale_x
    y2_new = y2 * scale_y
    return [x1_new, y1_new, x2_new, y2_new]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    Boxes are in format [x_min, y_min, x_max, y_max].
    
    Args:
        box1: List of 4 floats representing the first bounding box
        box2: List of 4 floats representing the second bounding box
    
    Returns:
        float: IoU value between 0 and 1
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Calculate intersection area
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Calculate areas of both boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate union area
    union_area = box1_area + box2_area - inter_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def evaluate_grounding(
    predictions: List[float],
    ground_truths: List[float],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate visual grounding predictions against ground truth boxes.
    
    Args:
        predictions: List of float containing predicted boxes in format [x_min, y_min, x_max, y_max]
        ground_truths: List of dicts containing ground truth boxes in same format
        iou_threshold: Float, IoU threshold for considering a match (default: 0.5)
    
    Returns:
        Dict containing precision, recall, and F1 score
    """

    flag= False
    if predictions: # If prediction is not empty
        iou = calculate_iou(predictions, ground_truths)
        if iou >= iou_threshold:
            flag = True
    
    return flag

def get_entity_popularity_cnt(wikipedia_url_list: list[str], start: str='20250101', end: str='20251231') -> dict[str, int]:
    # https://github.com/mediawiki-utilities/python-mwviews
    client = PageviewsClient(user_agent="<hanseok.oh@mila.quebec> Wikipedia entity popularity analysis")
    ent_name_list = [wikipedia_url.split("/")[-1] for wikipedia_url in wikipedia_url_list]
    views = client.article_views('en.wikipedia',ent_name_list , start=start, end=end)
    ent_popularity_cnt = defaultdict(int)
    for timestamp, info_dict in views.items():
        for ent_name, view_cnt in info_dict.items():
            if view_cnt:
                ent_popularity_cnt[ent_name]+= view_cnt
        
    return dict(ent_popularity_cnt)


def calculate_distance(image_size, roi_bbox):
    """
    Calculate the Euclidean distance from the center of an image to the center of a region of interest (ROI).
    
    Parameters:
    image_size (list or tuple): Image dimensions [width, height]
    roi_bbox (list or tuple): Region of interest bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
    float: Euclidean distance in pixels
    """
    # Calculate image center
    image_center_x = image_size[0] / 2
    image_center_y = image_size[1] / 2
    
    # Calculate ROI center
    roi_center_x = (roi_bbox[0] + roi_bbox[2]) / 2
    roi_center_y = (roi_bbox[1] + roi_bbox[3]) / 2
    
    # Calculate Euclidean distance
    distance = ((image_center_x - roi_center_x) ** 2 + (image_center_y - roi_center_y) ** 2) ** 0.5

    # Calculate half of image diagonal
    image_diagonal = (image_size[0] ** 2 + image_size[1] ** 2) ** 0.5 / 2
    
    # Normalize distance
    normalized_distance = distance / image_diagonal if image_diagonal != 0 else 0
    
    return normalized_distance