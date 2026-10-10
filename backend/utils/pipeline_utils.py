import cv2 
import os 
import json 
import shutil 
import time 
from collections import defaultdict
import re 
import base64
from unidecode import unidecode
from Levenshtein import distance
from PIL import Image 
from copy import deepcopy 
import traceback 
import numpy as np 
import requests
from utils.config import gen_cfg, dev_cfg
from utils.insp_logger import user_logger, dev_logger 
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import paddle


class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  

    def forward(self, x):
        return self.backbone(x)

def load_model(model, load_path):

    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    model.eval()
    return model

def load_prototypes(prototypes_path):
    return torch.load(prototypes_path, map_location='cpu')


def classify_image(model, prototypes, image, threshold=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0) 
    model.eval()
    with torch.no_grad():
        embedding = model(image)
        distances = {label: torch.dist(embedding, prototype).item() for label, prototype in prototypes.items()}
        max_distance = max(distances.values())
        percentage_distances = {label: (1 - (distance / max_distance)) * 100 for label, distance in distances.items()}
        closest_label = max(percentage_distances, key=percentage_distances.get)
        closest_distance = percentage_distances[closest_label]
        if closest_distance < threshold:
            closest_label = "Unidentified symbol"
    return closest_label, percentage_distances

def extract_numeric_value(string):
    pattern = r'\d+'
    match = re.search(pattern, string)
    if match:
        return int(match.group())
    else:
        return None
    
def form_pairs(tuples_list, string):
    pairs_dict = {}
    for i in range(len(tuples_list)):
        start_index = tuples_list[i][1]
        end_index = tuples_list[i + 1][1] if i < len(tuples_list) - 1 else len(string)
        pairs_dict[tuples_list[i][0]] = string[start_index:end_index]
    return pairs_dict

def get_possible_langs(metadata):
    fc_langs     = metadata["fc_langs"]
    care_langs   = metadata["care_langs"]
    excare_langs = metadata["excare_langs"]
    COO_langs    = metadata["COO_langs"]
    # fc_langs     = metadata["fc_langs"].split('/')
    # care_langs   = metadata["care_langs"].split('/')
    # excare_langs = metadata["excare_langs"].split('/')
    # COO_langs    = metadata["COO_langs"].split('/')
    return list(set(fc_langs+care_langs+excare_langs+COO_langs))

def dark_ratio(image, threshold=dev_cfg['default_dark_ratio_thresh']):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    dark_pixels = cv2.countNonZero(binary_image)
    total_pixels = binary_image.size
    percentage_dark = (dark_pixels / total_pixels) * 100
    
    return percentage_dark

def encode_file(file_path, file_type = '.pdf'):
    if file_type == '.pdf': 
        encoded_pdf = base64.b64encode(open(file_path, 'rb').read())
        import pdb; pdb.set_trace()


def format_material_info(material_info):
    # Get all unique materials
    materials = set()
    for category in material_info:
        materials.update(material_info[category].keys())

    # Create the header row
    header_row = ['Material'] + list(materials)

    # Create the data rows
    data_rows = [header_row]
    for category in material_info:
        row = [category]
        for material in materials:
            if material in material_info[category]:
                row.append(material_info[category][material])
            else:
                row.append(None)
        data_rows.append(row)

    return data_rows

def plot_material_info(material_info):
    # Format the data
    data = format_material_info(material_info)
    header = data[0]
    rows = data[1:]

    # Calculate column widths
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*data)]

    # Calculate image size
    img_height = (len(data) + 1) * 30
    img_width = sum(col_widths) * 20

    # Create white background image
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Add text to the image
    y = 30  # Starting y-coordinate
    for row in data:
        x = 10  # Starting x-coordinate
        for cell, width in zip(row, col_widths):
            cv2.putText(img, str(cell), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            x += width * 20  # Move to the next cell
        y += 30  # Move to the next row
    return img 

def decode_and_save(input_str, path):
    try:
        decoded = base64.b64decode(input_str)
        decoded_file = open(path, 'wb')
        decoded_file.write(decoded)
        decoded_file.close()
        return 0
    except Exception as e:
        ext = input_str.split('/')[-1].split('.')[-1]
        defect_str = f"ERROR: Incorrect .{ext} file, please ensure correct inputs and try again."
        return send_pipeline_fail(defect_str)

def encode_img(img):
    _, img_bytes = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64 

def decode_img(img_base64):
    img_bytes_decoded = base64.b64decode(img_base64)
    img_decoded = cv2.imdecode(np.frombuffer(img_bytes_decoded, np.uint8), cv2.IMREAD_COLOR)
    return img_decoded 

def encode_str(input_string):
    input_bytes = input_string.encode('utf-8')
    base64_bytes = base64.b64encode(input_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

def decode_str(encoded_string):
    base64_bytes = encoded_string.encode('utf-8')  
    decoded_bytes = base64.b64decode(base64_bytes)
    decoded_string = decoded_bytes.decode('utf-8')
    return decoded_string

def save_json(input_dict, path=None):

    
    with open(path, 'w') as json_file:
        json.dump(input_dict, json_file, indent=4)

def send_preprocessed_inputs(gt, url=None, search_mode = False):

    if search_mode:
        preprocessed = {
        'fiber_content'
    }
        fc_gt_dict = deepcopy(gt['fc_gt'])
        fc_gt = fc_gt_dict
        fc_gt_transformed = []
        headers = list(fc_gt.keys())
        fc_gt_transformed.append(headers)
        
        # Find the maximum length of the lists in fc_gt
        max_length = max(len(lst) for lst in fc_gt.values())
        
        # Fill in the shorter lists with None to match the max length
        for key in fc_gt:
            while len(fc_gt[key]) < max_length:
                fc_gt[key].append(None)
        
        rows = zip(*fc_gt.values())
        fc_gt_transformed.extend(list(rows))
        pdf_imgs = gt['pdf_image']

        pdf_imgs_encoded = []
        for pdf_img in pdf_imgs:
            encoded_img = encode_img(pdf_img)
            pdf_imgs_encoded.append(encoded_img)

            # pdf_imgs_encoded.append('abc')
        table_info = {
            'keyword': fc_gt_transformed,
            # 'care_instr': [], 
            # 'excare_instr': [], 
            # 'symbols': [],
            'pdf': pdf_imgs_encoded
        }
        # with open("preprocessed_data.json", "w") as json_file:
        #     json.dump(table_info, json_file)

        try:
            if gen_cfg['debug_endpoints']:
                import pdb; pdb.set_trace() 

            address = f"{gen_cfg['protocol']}://{gen_cfg['frontend_domain']}:{gen_cfg['frontend_port']}/api/inspection/Preprocessed"
            headers = {"X-Api-Key": gen_cfg['api_key'], "Content-Type": "application/json"}
            response = requests.post(address, json = table_info, headers = headers, verify = True, timeout=30)# Verificatio for SSL certificate
            
            if response.status_code == 200:
                dev_logger.info(f"INFO: Preprocessed Response: SUCCESS - {response.status_code}")
            else:
                user_logger.info(f"INFO: Something went wrong while processing the input files, please ensure correct network details.")
                dev_logger.info(f"INFO: Preprocessed Response: FAILED - {response.status_code}")

            return table_info
        except:
            return table_info
    else:
        preprocessed = {
            'fiber_content'
        }
        
        fc_gt = gt['fc_gt']
        if 'GB' in gt['care_gt'].keys():
            care_gt = gt['care_gt']['GB']
        elif 'UK' in gt['care_gt'].keys():
            care_gt = gt['care_gt']['UK']   
        elif 'US' in gt['care_gt'].keys():
            care_gt = gt['care_gt']['US']   
        else:
            care_gt = []
            user_logger.info(f"WARNING: Could not find any care instructions in the ground truth for GB, UK, or US. Tables will show empty.")
        
        if 'GB' in gt['excare_gt'].keys():
            excare_gt = gt['excare_gt']['GB']
        elif 'UK' in gt['excare_gt'].keys():
            excare_gt = gt['excare_gt']['UK']
        elif 'US' in gt['excare_gt'].keys():
            excare_gt = gt['excare_gt']['US']
        else:
            excare_gt = []
            dev_logger.info( f"WARNING: Could not find any extra care instructions in the ground truth for GB, UK, or US. Tables will show empty.")


        if 'GB' in gt['fc_gt'].keys():
            fc_gt = gt['fc_gt']['GB']
        elif 'UK' in gt['fc_gt'].keys():
            fc_gt = gt['fc_gt']['UK']
        elif 'US' in gt['fc_gt'].keys():
            fc_gt = gt['fc_gt']['US']
        else:
            fc_gt = []
            user_logger.info(f"WARNING: Could not find any fiber content in the ground truth for GB, UK, or US. Tables will show empty.")
            dev_logger.info( f"WARNING: Could not find any fiber content in the ground truth for GB, UK, or US. Tables will show empty.")


        symbols_gt = gt['symbols_gt']
        symbols_gt_refined = {key.replace('*', ''):{} for key in symbols_gt.keys()}
        for symbol_type, symbol_dict in symbols_gt.items():
            symbol_type = symbol_type.replace('*', '')
            filename = os.listdir(os.path.join(gen_cfg['symbols_db'], symbol_dict['symbol_id']))[0]
            folder = os.path.join(gen_cfg['symbols_db'], symbol_dict['symbol_id'])
            symbol_img = cv2.resize(cv2.imread(os.path.join(folder, filename)), (50,50))
            symbols_gt_refined[symbol_type]['symbol_id'] = symbol_dict['symbol_id']
            symbols_gt_refined[symbol_type]['symbol'] = encode_img(symbol_img)

        col1 = list(symbols_gt_refined.keys())
        col2 = [symbol_dict['symbol_id'] for symbol_dict in list(symbols_gt_refined.values())]
        col3 = [symbol_dict['symbol'] for symbol_dict in list(symbols_gt_refined.values())]
        symbols_gt_refined = [col1, col2, col3]

        pdf_imgs = gt['pdf_image']

        # symbols_gt_refined = {}
        # for symbol_key, symbol_dict in symbols_gt.items(): 
        #     id = symbol_dict['symbol_id']
        #     instr = symbol_dict['symbol_instr']
        #     symbols_gt_refined[symbol_key.replace('*','')] = f'{id} - {instr}'
        fc_gt_refined = format_material_info(fc_gt)

        pdf_imgs_encoded = []
        for pdf_img in pdf_imgs:
            encoded_img = encode_img(pdf_img)
            pdf_imgs_encoded.append(encoded_img)

            # pdf_imgs_encoded.append('abc')
        table_info = {
            'fiber_content': fc_gt_refined,
            'care_instr': care_gt, 
            'excare_instr': excare_gt, 
            'symbols': symbols_gt_refined,
            'pdf': pdf_imgs_encoded
        }
        # with open("preprocessed_data.json", "w") as json_file:
        #     json.dump(table_info, json_file)

        try:
            if gen_cfg['debug_endpoints']:
                import pdb; pdb.set_trace() 

            address = f"{gen_cfg['protocol']}://{gen_cfg['frontend_domain']}:{gen_cfg['frontend_port']}/api/inspection/Preprocessed"
            headers = {"X-Api-Key": gen_cfg['api_key'], "Content-Type": "application/json"}
            response = requests.post(address, json = table_info, headers = headers, verify = True, timeout=30)
            
            if response.status_code == 200:
                dev_logger.info(f"INFO: Preprocessed Response: SUCCESS - {response.status_code}")
            else:
                user_logger.info(f"INFO: Something went wrong while processing the input files, please ensure correct network details.")
                dev_logger.info(f"INFO: Preprocessed Response: FAILED - {response.status_code}")

            return table_info
        except:
            return table_info

def remove_close_boxes(article_langs, bbox_dist_thresh=dev_cfg['default_bbox_remove_thresh']):
    cleaned_article_langs = {}
    for lang, bboxes in article_langs.items():
        cleaned_bboxes = []
        if len(bboxes) > 1:
            prev_ymin = bboxes[0][1]
            cleaned_bboxes.append(bboxes[0])
            for bbox in bboxes[1:]:
                ymin = bbox[1]
                if ymin - prev_ymin >= bbox_dist_thresh:
                    cleaned_bboxes.append(bbox)
                    prev_ymin = ymin
        else:
            cleaned_bboxes = bboxes
        cleaned_article_langs[lang] = cleaned_bboxes
    return cleaned_article_langs

def sort_lang_dict(article_langs):
    list_to_sort = []

    for lang, bbox_list in article_langs.items():
        for bbox in bbox_list: 
            new_bbox = bbox+[lang]
            list_to_sort.append(new_bbox)
    
    sorted_list = sorted(list_to_sort, key=lambda x:x[1])

    return sorted_list

def remove_contents(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


def remove_empty_lists(dictionary):
    return {key: value for key, value in dictionary.items() if value}

def get_biggest_page_size(pages_filtered):
    biggest_h, biggest_w = pages_filtered[1][0].shape[:2]

    for page_num, page_list in pages_filtered.items():
        for page in page_list: 
            curr_h, curr_w = page.shape[:2]
            if curr_h > biggest_h:
                biggest_h = curr_h 
            if curr_w > biggest_w: 
                biggest_w = curr_w 
    
    return (biggest_h, biggest_w)

def get_page_number(height, n, k):
    range_width = height / n

    for i in range(n):
        start = i * range_width
        end = (i + 1) * range_width
        if start <= k < end:
            return i
    
    if k == height:
        return n - 1

    return -1

def cut_layout_pages(pages, scissors_model):
    try: 
        symbols = {}
        pages_without_scissors = {}
        for page_num , page_crop in pages.items():
            scissors_det = scissors_model(page_crop,conf=dev_cfg["yolo_conf"], verbose=False) 
            scissors_ymin, scissors_ymax = None, None
            symbols[page_num] = []
            for result in scissors_det: 
                boxes = result.boxes 
                for cls_idx, cls in enumerate(boxes.cls): 
                    if int(cls.item()) == 0:
                        use_box = boxes.xyxy.tolist()[cls_idx]
                        scissors_ymin = int(use_box[1])
                        scissors_ymax = int(use_box[3])
                    elif int(cls.item()) == 1:
                        symbol_bbox = boxes.xyxy.tolist()[cls_idx]
                        symbol_bbox = [int(elem) for elem in symbol_bbox]
                        symbol_crop = page_crop[symbol_bbox[1]:symbol_bbox[3], symbol_bbox[0]:symbol_bbox[2]]
                        symbol_dict = {'bbox':symbol_bbox, 'crop':symbol_crop}
                        symbols[page_num].append(symbol_dict)
                    elif int(cls.item()) == 2:
                        dev_logger.info(f"INFO: Korean symbols detected on Page #{page_num}, ignoring.")
            # if page_num % 2: # odd page, scissors is on top 
            #     # confirm its within the top quadrant 
            if scissors_ymax is not None and scissors_ymin is not None:
                top_quad = scissors_ymax <= page_crop.shape[0]//4
                dev_logger.info(f'INFO: Found scissors in top quadrant: {top_quad}')
                if top_quad:
                    page_without_scis = page_crop[scissors_ymax:, :]
                else:
                    page_without_scis = page_crop[0:scissors_ymin, :]
                # else:
                #     page_without_scis = page_crop[0:scissors_ymin, :]
            else:
                page_without_scis = page_crop

            if not dark_ratio(page_without_scis) < 5:
                pages_without_scissors[page_num] = page_without_scis

        return pages_without_scissors, symbols
    except: 
        defect_str = f"ERROR: Invalid PDF configuration, something went wrong in stack creation. Please check your inputs / see inner exception for more details."
        return send_pipeline_fail(defect_str)

def stack_layout(pages):
    sorted_pages = [pages[key] for key in sorted(pages)]
    
    import pdb; pdb.set

def cut_pages(detector_model, pages_filtered, ocr, item_name):
    try:
        lower_ys_odd = []
        higher_ys_odd = []

        lower_ys_even = []
        higher_ys_even = []

        symbols = {}
        for page_num, pages in pages_filtered.items(): 
            symbols[page_num] = []
            if not page_num % 2: # even num pages (top of the PDF)
                for idx, page_uncropped in enumerate(pages):
                    page_h, page_w = page_uncropped.shape[:2]
                    results = detector_model(page_uncropped,conf=dev_cfg["yolo_conf"], verbose=False) 
                    scissors_y = None
                    for result in results: 
                        boxes = result.boxes 
                        for cls_idx, cls in enumerate(boxes.cls): 
                            if int(cls.item()) == 0:
                                scissors_y = int(boxes.xyxy.tolist()[cls_idx][1])
                            elif int(cls.item()) == 1:
                                symbol_bbox = boxes.xyxy.tolist()[cls_idx]
                                symbol_bbox = [int(elem) for elem in symbol_bbox]
                                symbol_crop = page_uncropped[symbol_bbox[1]:symbol_bbox[3], symbol_bbox[0]:symbol_bbox[2]]
                                symbol_dict = {'bbox':symbol_bbox, 'crop':symbol_crop}
                                symbols[page_num].append(symbol_dict)
                    # cut the top scissors part
                    # lower_y = mm_to_px(6.5)
                    # cv2.imwrite(f'/home/visionrd/avery/pipeline/data/scissorz/{page_num}_{idx}.png', page_uncropped)
                    if scissors_y is not None: 
                        page = page_uncropped[0:scissors_y, :]
                        lower_ys_even.append(scissors_y)
                    else: 
                        dev_logger.info(f"Could not find a scissors in Page {page_num}, instance {idx}")
                        page = page_uncropped[0:page_uncropped.shape[0], :]
                        lower_ys_even.append(page.shape[0])
                    try: 
                        top_quarter = page[0:page.shape[0]//4, :]
                        torch.cuda.empty_cache()
                        paddle.device.cuda.empty_cache()
                        top_quarter_text = ocr.ocr(top_quarter, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                        torch.cuda.empty_cache()
                        paddle.device.cuda.empty_cache()
                        page_bbox = [element for sublist in top_quarter_text for element in sublist if 'Page' in element[1][0]]
                        
                    except:
                        defect_str = f"ERROR: Invalid PDF configuration, could not find any text in top quarter of Page {page_num} while cutting, instance {idx}"
                        return send_pipeline_fail(defect_str)


                    xmin, ymin = None, None 
                    if page_bbox:
                        if page_bbox[0] is not None: 
                            try: 
                                xmin = int(page_bbox[0][0][0][0])
                                ymin = int(page_bbox[0][0][0][1])
                            except:
                                defect_str = f"ERROR: Invalid PDF configuration. Could not find  keyword 'Page' in even pages, index: {idx}. Failed to stack."
                                return send_pipeline_fail(defect_str)

                                # os.makedirs(f'results/{item_name}/errors/', exist_ok=True)
                                # cv2.imwrite(f'results/{item_name}/errors/stack_{idx}.png',page)

                    if xmin is not None and ymin is not None: 
                        word_height = dev_cfg['even_word_height']
                        stride = dev_cfg['even_cut_stride']

                        # first find the first completely white (empty) strip
                        y = ymin 
                        strip = page[y:y+word_height, :]
                        while dark_ratio(strip) != 100: 
                            y += stride 
                            strip = page[y:y+word_height, :]

                        count = 0

                        while y <= page.shape[0]:

                            # strip_top = max(0, y)
                            # strip_bottom = min(y + word_height, ymin)
                            strip = page[y:y+word_height, :]
                            torch.cuda.empty_cache()
                            paddle.device.cuda.empty_cache()
                            strip_text = ocr.ocr(strip, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
                            torch.cuda.empty_cache()
                            paddle.device.cuda.empty_cache()
                            if strip_text[0] is not None or dark_ratio(strip) < dev_cfg['strip_dark_ratio_thresh']: 
                                higher_ys_even.append(y)
                                # os.makedirs(f'results/{item_name}/stack_strips', exist_ok=True)
                                # cv2.imwrite(f'results/{item_name}/stack_strips/strip_{page_num}_{idx}.png', strip)
                                break
                            y += stride
                            # cv2.imwrite(f'results/strips/{page_num}_{idx}_{count}.png', strip )
                            count+=1
                    pages_filtered[page_num][idx] = page 
            else:  # odd num pages (bottom of the PDF)
                
                for idx, page_uncropped in enumerate(pages):
                    page_h, page_w = page_uncropped.shape[:2]
                    results = detector_model(page_uncropped, conf= dev_cfg['yolo_conf'], verbose=False) 
                    scissors_y = None
                    for result in results: 
                        boxes = result.boxes 
                        
                        for cls_idx, cls in enumerate(boxes.cls): 
                            if int(cls.item()) == 0:
                                scissors_y = int(boxes.xyxy.tolist()[cls_idx][-1])
                            elif int(cls.item()) == 1:
                                symbol_bbox = boxes.xyxy.tolist()[cls_idx]
                                symbol_bbox = [int(elem) for elem in symbol_bbox]
                                symbol_crop = page_uncropped[symbol_bbox[1]:symbol_bbox[3], symbol_bbox[0]:symbol_bbox[2]]
                                symbol_dict = {'bbox':symbol_bbox, 'crop':symbol_crop}
                                symbols[page_num].append(symbol_dict)
                    # cut the top scissors part
                    # cv2.imwrite(f'/home/visionrd/avery/pipeline/data/scissorz/{page_num}_{idx}.png', page_uncropped)
                    
                    if scissors_y is not None: 
                        page = page_uncropped[scissors_y:page_uncropped.shape[0], :]
                        higher_ys_odd.append(scissors_y)
                    else: 
                        dev_logger.info(f"WARNING: Could not find a scissors in Page {page_num}, instance {idx}")
                        
                        page = page_uncropped[0:page_uncropped.shape[0], :]
                        higher_ys_odd.append(page.shape[0])
                    # find 'Page' in bottom quarter
                    try:

                        bottom_quarter = page[3*page.shape[0]//4:page.shape[0], :]
                        torch.cuda.empty_cache()
                        paddle.device.cuda.empty_cache()
                        bottom_quarter_text = ocr.ocr(bottom_quarter, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                        torch.cuda.empty_cache()
                        paddle.device.cuda.empty_cache()
                        
                        page_bbox = [element for sublist in bottom_quarter_text for element in sublist if 'Page' in element[1][0]]
                    except:
                        continue 
                    xmin, ymin = None, None 
                    if page_bbox:
                        if page_bbox[0] is not None: 
                            try: 
                                xmin = int(page_bbox[0][0][0][0])
                                ymin = int(3*page.shape[0]//4+page_bbox[0][0][0][1])
                            except:
                                defect_str = f"ERROR: Invalid PDF configuration. Could not find  keyword 'Page' in odd pages, index: {idx}. Failed to stack."
                                return send_pipeline_fail(defect_str)
                                
                                # os.makedirs(f'results/{item_name}/errors/', exist_ok=True)
                                # cv2.imwrite(f'results/{item_name}/errors/stack_{idx}.png',page)
                    if xmin is not None and ymin is not None: 
                        word_height = dev_cfg['odd_word_height']
                        stride = dev_cfg['odd_cut_stride']
                        
                        y = ymin - word_height
                        count = 0
                        while y >= 0:
                            strip_top = max(0, y-word_height)
                            strip_bottom = min(y, page.shape[0])

                            strip = page[strip_top:strip_bottom, :]
                            torch.cuda.empty_cache()
                            paddle.device.cuda.empty_cache()
                            strip_text = ocr.ocr(strip, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
                            torch.cuda.empty_cache()
                            paddle.device.cuda.empty_cache()
                            if strip_text[0] is not None or dark_ratio(strip) < dev_cfg['strip_dark_ratio_thresh']:
                                lower_ys_odd.append(strip_bottom)
                                # os.makedirs(f'results/{item_name}/stack_strips', exist_ok=True)
                                # cv2.imwrite(f'results/{item_name}/stack_strips/strip_{page_num}_{idx}.png', strip)
                                break
                            
                            y -= stride
                            # cv2.imwrite(f'results/strips/{page_num}_{idx}_{count}.png', strip )
                            count+=1
                    pages_filtered[page_num][idx] = page 

        ymax_odd = max(lower_ys_odd)
        ymax_even = max(lower_ys_even)
        ymin_even = min(higher_ys_even)
        new_dict = {}
        for page_num, pages in pages_filtered.items():
            new_dict[page_num] = []
            if not page_num % 2:
                for page_num_num, page in enumerate(pages):
                    # cv2.imwrite(f'/home/visionrd/avery/pipeline/data/scissorz/{page_num}_{page_num_num}.png', page)
                    new_page = page[ymin_even:ymax_even, :]
                    new_dict[page_num].append(new_page)
                    # os.makedirs(f'results/{item_name}/pages_cropped/Page_{page_num}/', exist_ok=True)
                    # cv2.imwrite(f'results/{item_name}/pages_cropped/Page_{page_num}/{page_num_num}.png',new_page)
            else:
                for page_num_num, page in enumerate(pages):

                    new_page = page[0:ymax_odd, :]
                    new_dict[page_num].append(new_page)
                    # os.makedirs(f'results/{item_name}/pages_cropped/Page_{page_num}/', exist_ok=True)
                    # cv2.imwrite(f'results/{item_name}/pages_cropped/Page_{page_num}/{page_num_num}.png',new_page)
        return new_dict, symbols
    
    except Exception as e: 
        defect_str = f"ERROR: Invalid PDF configuration, something went wrong in stack creation. Please check your inputs / see inner exception for more details."
        return send_pipeline_fail(defect_str)
    

def get_biggest_stack(horizontal_stacks):
    biggest_h, biggest_w = horizontal_stacks[0].shape[:2]

    for stack in horizontal_stacks:
        curr_h, curr_w = stack.shape[:2]
        if curr_h > biggest_h:
            biggest_h = curr_h 
        if curr_w > biggest_w: 
            biggest_w = curr_w 
    
    return (biggest_h, biggest_w)


def find_str_in_gt(stringA, stringB, return_indices=True, diff_thesh=dev_cfg['default_str_thresh']):
    stringA = stringA.upper()
    stringB = stringB.upper()
    if stringA in stringB:
        if return_indices:
            start_index = stringB.index(stringA)
            diff = distance(stringA, stringB[start_index:start_index+len(stringA)])
            return True, start_index, diff
        else:
            return True
    else:
        if return_indices: 
            found, found_indices, min_diff = find_substringwise(stringA, stringB, return_indices=return_indices, diff_thesh=diff_thesh)
            return found, found_indices, min_diff
        
        else:
            found = find_substringwise(stringA, stringB, return_indices=return_indices)
            return found, min_diff
        

def find_substringwise(A, B, return_indices=True, diff_thesh=dev_cfg['default_str_thresh']):
    min_diff = float('inf')  # Initialize min_diff to positive infinity
    min_index = None  # Initialize min_index to None
    
    for i in range(len(B) - len(A) + 1):
        compare_b = B[i:i+len(A)]
        diff_count = distance(compare_b, A)
        
        if diff_count < min_diff:
            min_diff = diff_count
            min_index = i
        
        if diff_count == 0:  # If exact match found, no need to continue searching
            break
    
    if min_diff < diff_thesh:  # If minimum difference is less than 3
        if return_indices:
            return True, min_index, min_diff
        else:
            return True, min_index, min_diff  # Return both flag and index
    else:
        if return_indices:
            return False, None, min_diff
        else:
            return False, min_diff
        

def return_all_strings(text):
    null = ''
    for txt in text[0]: 
        null += txt[-1][0]
    return null

def find_fc_idx(series1, series2, target, max_distance=3):
    idxs1 = {}
    for i, value in series1.items():
        dist = distance(value, target)
        if  dist < max_distance:
            idxs1[i] = dist

    idxs2 = {}
    for i, value in series2.items():
        dist = distance(value, target)
        if  dist < max_distance:
            idxs2[i] = dist 
    combined_idxs = {}
    for i, dist in idxs1.items():
        combined_idxs[(i, 'eu')] = dist
    for i, dist in idxs2.items():
        combined_idxs[(i, 'us')] = dist
    
    if combined_idxs:
        sorted_idxs = sorted(combined_idxs.items(), key=lambda item: (item[1], item[0][1] == 'us'))
        return sorted_idxs[0][0][0], sorted_idxs[0][1]
    else:
        return None, None

def insert_integer(container_key, joined_container_key, number):
    # Find where the strings start to differ
    diff_index = 0
    min_len = min(len(container_key), len(joined_container_key))
    for i in range(min_len):
        if container_key[i] != joined_container_key[i]:
            diff_index = i
            break

    result = joined_container_key[:diff_index] + ' [{}]'.format(number) + joined_container_key[diff_index:]
    return result

def check_if_recycling_perc(s):
    # Use regex to find all substrings within parentheses
    matches = re.findall(r'\((.*?)\)', s)
    # Check if any of the matched substrings contain numeric values
    for match in matches:
        if re.search(r'\d', match):
            return True
    return False

def remove_special_chars(input_string):
    # Use regular expression to match only alphabets and numbers
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return cleaned_string
def inspect_fiber_content(fiber_content, fiber_content_gt, blacklisted_langs, defects, use_decode=True):

    exclusive = False 
    material_diff_thresh = dev_cfg['material_diff_thresh']

    # first get rid of languages that are not supposed to be checked 
    missing_langs = blacklisted_langs['missing']
    extra_langs = blacklisted_langs['extra']
    unsupported_langs = blacklisted_langs['unsupported']
    blacklisted = missing_langs | extra_langs | unsupported_langs 
    if missing_langs: 
        user_logger.info(f"WARNING: Missing languages in fiber content: {missing_langs}")
        dev_logger.info( f"WARNING: Missing languages in fiber content: {missing_langs}")

    if extra_langs:
        user_logger.info(f"WARNING: Extra languages in fiber content: {extra_langs}")
        dev_logger.info( f"WARNING: Extra languages in fiber content: {extra_langs}")

    for lang in blacklisted: 
        if lang in fiber_content_gt.keys():
            user_logger.info(f"WARNING: {lang} blacklisted from fiber GT.")
            dev_logger.info(f"WARNING: {lang} blacklisted from fiber GT.")
            del fiber_content_gt[lang]
        if lang in fiber_content.keys():
            user_logger.info(f"WARNING: {lang} blacklisted from fiber GT.")
            dev_logger.info(f"WARNING: {lang} blacklisted from fiber GT.")
            del fiber_content[lang]

    defects_fc = {}
    defects_fc['total'] = 0
    for lang in fiber_content.keys():
        defects_fc[lang] = []

    # start inspection
    # for (gt_lang, gt_dict), (pdf_lang, pdf_dict) in zip(fiber_content_gt.items(), fiber_content.items()):
    for gt_lang, gt_dict in fiber_content_gt.items():
        user_logger.info(f"INFO: Inspecting fiber content in: {gt_lang}")
        dev_logger.info(f"INFO: Inspecting fiber content in: {gt_lang}")
        pdf_dict = fiber_content[gt_lang]
        pdf_text = ''
        for instance in pdf_dict:
            for lines in instance['text']:
                pdf_text += lines.replace(' ', '')

        if use_decode:
            pdf_text = unidecode(pdf_text)

        if not exclusive: 
            excl_found, excl_idx, excl_dist = find_str_in_gt('EXCLUSIVEOFTRIMMINGANDDECORATION', pdf_text, return_indices=True, diff_thesh=material_diff_thresh)
            if excl_found:
                exclusive = True 

        pdf_indices = []
        defective_material_types = []
        for material_type in gt_dict.keys():
            material_type = material_type.replace(' ', '')
            material_type_found, material_type_idx, material_type_dist = find_str_in_gt(material_type, pdf_text, return_indices=True, diff_thesh=material_diff_thresh)
            if material_type_found: 
                pdf_indices.append((material_type, material_type_idx)) 
            else:
                defects_fc['total']+=1
                defect_crop = fiber_content[gt_lang][0]['lang_crop']
                defect_page_num = fiber_content[gt_lang][0]['page_num']
                defect_str = f'DEFECT: Missing/incorrect part position {material_type} in PDF at Page #{defect_page_num}'
                defect_dict = {
                    "defect_str": defect_str, 
                    "defect_crop": defect_crop
                }
                defects_fc[gt_lang].append(defect_dict)

                user_logger.info(defect_str)
                dev_logger.info(defect_str)
                defective_material_types.append(material_type)

        # for material_type in defective_material_types:
        #     del gt_dict[material_type]

        pdf_indices_dict = form_pairs(pdf_indices, pdf_text)
        for material_type, material_dict in gt_dict.items():
            

            if use_decode:
                material_type_get = unidecode(material_type).replace(' ', '')
            else:
                material_type_get = material_type.replace(' ', '')

            if material_type_get in defective_material_types:
                continue 

            pdf_text = pdf_indices_dict[material_type_get].upper()
            for material, material_perc in material_dict.items():
                material_diff_thresh = dev_cfg['material_diff_thresh']
                look_back_failure = False 
                found_behind = False
                found_ahead = False 
                matches = {'material': False, 'percentage': False}
                # get rid of lang-specific modifications 
                if use_decode:
                    material = unidecode(material.replace(' ', '')).upper()
                else:
                    material = material.replace(' ', '').upper()

                if '(' in pdf_text or ')' in pdf_text:
                    pdf_text = pdf_text.replace('(', '').replace(')', '')
                if '(' in material or ')' in material:
                    material = material.replace('(', '').replace(')', '')
                    
                if gt_lang == 'US' and material == 'ELASTANE':
                    material = 'SPANDEX'
                # check type 1: find material str and lookback the percentage
                look_back = len(str(material_perc)) + 1 
                
                found, found_idx, material_dist = find_str_in_gt(material, pdf_text, return_indices=True, diff_thesh=material_diff_thresh)
                if found: 
                    material_end_idx = found_idx+len(material)-material_dist
                    material_found = pdf_text[found_idx:material_end_idx]
                    if distance(material, material_found) < material_diff_thresh:
                        matches['material'] = True 

                    try: 
                        material_perc_found = pdf_text[found_idx-look_back:found_idx]
                        material_perc_found = extract_numeric_value(material_perc_found.replace('%', ''))
                        material_perc_found = int(material_perc_found)
                    except Exception as wrong_pdf_string: 
                        user_logger.info(f'WARNING: {material} percentage in PDF is invalid: "{material_perc_found}"{wrong_pdf_string}{traceback.format_exc()}')
                        dev_logger.info(f'WARNING: {material} percentage in PDF is invalid: "{material_perc_found}"{wrong_pdf_string}{traceback.format_exc()}')
                        
                        look_back_failure = True 
                    if not look_back_failure: 
                        if material_perc == material_perc_found: 
                            matches['percentage'] = True 
                    else: 
                        dev_logger.info(f"WARNING: Could not find material ({material}) percentage with lookback method, attempting back and forth method")
                        
                        # check type 2: find percentage then find material 
                        material_perc_str = str(material_perc) + '%'
                        found_perc, found_perc_idx, perc_dist = find_str_in_gt(material_perc_str, pdf_text, return_indices=True, diff_thesh=material_diff_thresh)
                        if found_perc: 
                            found_perc_str = pdf_text[found_perc_idx:found_perc_idx+len(material_perc_str)]
                            
                            if material_perc_str == found_perc_str:
                                material_perc_found = int(extract_numeric_value(found_perc_str.replace('%', '')))
                                matches['percentage'] = True 

                            if not matches['material']: 
                                # try to find material behind the percentage 
                                material_found = pdf_text[found_perc_idx-len(material):found_perc_idx] 
                                if distance(material_found, material) < material_diff_thresh: 
                                    dev_logger.info(f'Found {material_found} behind percentage.')
                                    found_behind=True 
                                    matches['material'] = True 

                                # try to find material ahead of the percentage 
                                if not found_behind: 
                                    material_found = pdf_text[found_perc_idx+look_back:found_perc_idx+look_back+len(material)]
                                    if distance(material_found, material) < material_diff_thresh: 
                                        dev_logger.info(f'Found {material_found} ahead of percentage.')
                                        found_ahead = True
                                        matches['material'] = True 

                if matches['material']:
                    user_logger.info(f"MATCHED: {material} (EXCEL) == {material_found} (PDF) in {material_type}")   
                    dev_logger.info(f"MATCHED: {material} (EXCEL) == {material_found} (PDF) in {material_type}")   
                    
                    if not look_back_failure: 
                        # blankout material 
                        pdf_text = pdf_text[:found_idx] + '#' * len(pdf_text[found_idx:material_end_idx]) + pdf_text[material_end_idx:]
                    else: 
                        if found_behind: 
                            pdf_text = pdf_text[:found_perc_idx-len(material)] + '#' * len(pdf_text[found_perc_idx-len(material):found_perc_idx]) + pdf_text[found_perc_idx:]
                        elif found_ahead: 
                            pdf_text = pdf_text[:found_perc_idx+look_back] + '#' * len(pdf_text[found_perc_idx+look_back:found_perc_idx+look_back+len(material)]) + pdf_text[found_perc_idx+look_back+len(material):]
                        else: 
                            pdf_text = pdf_text[:found_idx] + '#' * len(pdf_text[found_idx:material_end_idx]) + pdf_text[material_end_idx:]
                else: 
                    defects_fc['total'] += 1
                    defect_page_num = fiber_content[gt_lang][0]['page_num']
                    defect_str = f'DEFECT: Missing/incorrect material {material} in part position {material_type} at Page #{defect_page_num}'
                    defect_crop = fiber_content[gt_lang][0]['lang_crop']
                    defect_dict = {
                            "defect_str": defect_str, 
                            "defect_crop": defect_crop
                        }
                    defects_fc[gt_lang].append(defect_dict)
                    user_logger.info(defect_str)
                    dev_logger.info(defect_str)

                if matches['percentage']: 
                    user_logger.info(f"MATCHED: {material_perc}% (EXCEL) == {material_perc_found}% (PDF) in {material_type}")
                    dev_logger.info(f"MATCHED: {material_perc}% (EXCEL) == {material_perc_found}% (PDF) in {material_type}")
                    
                    if not look_back_failure:
                        pdf_text = pdf_text[:found_idx-look_back] + '#' * len(pdf_text[found_idx-look_back:found_idx]) + pdf_text[found_idx:]
                    else: 
                        pdf_text = pdf_text[:found_perc_idx] + '#' * len(pdf_text[found_perc_idx:found_perc_idx+len(material_perc_str)]) + pdf_text[found_perc_idx+len(material_perc_str):] 
   
                else: 
                    if matches['material']:
                        defects_fc['total'] += 1
                        defect_page_num = fiber_content[gt_lang][0]['page_num']
                        try:
                            defect_str = f'DEFECT: Missing/incorrect {material} percentage: {material_perc_found}% (PDF) != {material_perc}% (EXCEL) in {material_type} at Page #{defect_page_num}'
                        except:
                            defect_str = f'DEFECT: Missing/incorrect {material} percentage: {None}% (PDF) != {material_perc}% (EXCEL) in {material_type} at Page #{defect_page_num}'

                        defect_crop = fiber_content[gt_lang][0]['lang_crop']
                        defect_dict = {
                                "defect_str": defect_str, 
                                "defect_crop": defect_crop
                            }
                        defects_fc[gt_lang].append(defect_dict)

                        user_logger.info(defect_str)
                        dev_logger.info(defect_str)

                        
    
    defects_fc_refined = remove_duplicate_defects(defects_fc)
    defects['fiber_content'] = defects_fc_refined
    defects['total_defects'] += defects_fc['total']
    return defects, exclusive

def merge_sets(sets):
    merged = []
    while sets:
        first, *rest = sets
        first = set(first)

        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for s in rest:
                if first.intersection(s):
                    first |= set(s)
                else:
                    rest2.append(s)
            rest = rest2

        merged.append(first)
        sets = rest
    return merged


def remove_duplicate_defects(defect_dict):
    merged_dict = {key: [] for key in defect_dict if key != 'total'}

    def merge_defects_for_key(defect_list):
        if defect_list:
            for defect_idx, defect in enumerate(defect_list):
                crop = defect['defect_crop']
                defect_str = defect['defect_str']
                merge_idxs = []
                
                for defect_idx_2, defect_2 in enumerate(defect_list):
                    if defect_idx_2 == defect_idx:
                        continue 
                    crop_2 = defect['defect_crop']
                    defect_str_2 = defect['defect_str']
                    if np.array_equal(crop, crop_2): 
                        defect_str += ''+defect_str_2 
                        merge_idxs.append((defect_idx, defect_idx_2, defect_str))

            
            merged_list = []
            if merge_idxs:
                sets = merge_sets([set(idxs) for idxs in np.array(merge_idxs)[:,:2]])
                for curr_set in sets: 
                    defect_strs = []
                    for val in curr_set: 
                        defect_strs.append(defect_list[int(val)]['defect_str'])

                    defect_crop = defect_list[int(val)]['defect_crop']
                    defect_dict = {
                        'defect_crop':defect_crop, 
                        'defect_str':"\n".join(defect_strs)
                    }
                    merged_list.append(defect_dict)

            if merged_list:
                return merged_list
            else: 
                return defect_list
        
    for key in merged_dict.keys():
        merged_dict[key] = merge_defects_for_key(defect_dict[key])
    
    merged_dict['total'] = defect_dict['total']
    return merged_dict


def inspect_care(care_instr, care_instr_gt, extra_care_gt, blacklisted_langs, defects, use_decode=False, exclusive=False):

    missing_langs = blacklisted_langs['missing']
    extra_langs = blacklisted_langs['extra']
    unsupported_langs = blacklisted_langs['unsupported']
    blacklisted = missing_langs | extra_langs | unsupported_langs 

    for lang in blacklisted: 
        try: 
            if lang in care_instr_gt.keys():
                user_logger.info(f"{lang} blacklisted from care instruction GT.")
                dev_logger.info(f"{lang} blacklisted from care instruction GT.")
                del care_instr_gt[lang]

            if lang in care_instr.keys():
                user_logger.info(f"{lang} blacklisted from care instruction PDF.")
                dev_logger.info(f"{lang} blacklisted from care instruction PDF.")
                
                del care_instr[lang]

            if lang in extra_care_gt.keys():
                user_logger.info(f"{lang} blacklisted from extra care instruction GT.")
                dev_logger.info(f"{lang} blacklisted from extra care instruction GT.")
                
                del extra_care_gt[lang]
        except:
            continue 

    all_langs = set(list(care_instr_gt.keys()) + list(extra_care_gt.keys()) + list(care_instr.keys()))

    care_defects   = {lang: [] for lang in care_instr_gt.keys()} 
    excare_defects = {lang: [] for lang in extra_care_gt.keys()}

    care_defects['total'] = 0 
    excare_defects['total'] = 0

    care_gt_leftovers   = deepcopy(care_instr_gt)
    excare_gt_leftovers = deepcopy(extra_care_gt)
    
    care_idxs   = {lang: None for lang in care_instr_gt.keys()}
    excare_idxs = {lang: None for lang in extra_care_gt.keys()}

    for lang in all_langs: 
        
        try: 
            care_instr_langwise = care_instr_gt[lang]
        except: 
            care_instr_langwise = {} 
        
        try:
            excare_instr_langwise = extra_care_gt[lang]
        except:
            excare_instr_langwise = {} 
        
        try:
            pdf_text_langwise = care_instr[lang]
        except: 
            care_defects['total'] += 1
            excare_defects['total'] += 1
            defect_crop = np.ones((500,500,3), dtype=np.uint8) * 255
            defect_str = f'DEFECT: Missing Language For Care/Extra Care Instructions: {lang}'
            
            defect_dict = {
                'defect_str':defect_str, 
                'defect_crop':defect_crop
            }

            care_defects[lang].append(defect_dict) 
            excare_defects[lang].append(defect_dict) 

            dev_logger.info(defect_str)
            user_logger.info(defect_str)
            
            pdf_text_langwise = None 

            # this is a defect: missing lang 
        if pdf_text_langwise:
            user_logger.info(f"INFO: Inspecting care instructions in {lang}")
            dev_logger.info(f"INFO: Inspecting care instructions in {lang}")
            
            for text_idx, pdf_text in enumerate(pdf_text_langwise):
                pdf_text = pdf_text['text']
                if use_decode:
                    pdf_text = unidecode(pdf_text.replace(' ', ''))
                else: 
                    pdf_text = pdf_text.replace(' ', '')

                care_box, excare_box = False, False
                for care_instr_excel in care_instr_langwise:
                    
                    care_instr_excel_org = care_instr_excel 
                    if use_decode:
                        care_instr_excel = unidecode(care_instr_excel.replace(' ', ''))
                    else:
                        care_instr_excel = care_instr_excel.replace(' ', '')
                    pdf_text = remove_special_chars(pdf_text)
                    found_care, found_care_idx, care_dist = find_str_in_gt(care_instr_excel, pdf_text, return_indices=True)

                    if found_care: 
                        found_care_str = pdf_text[found_care_idx:found_care_idx+len(care_instr_excel)] 
                        user_logger.info(f'MATCHED: {care_instr_excel} (EXCEL) == {found_care_str} (PDF)')
                        dev_logger.info(f'MATCHED: {care_instr_excel} (EXCEL) == {found_care_str} (PDF)')
                        

                        care_gt_leftovers[lang].remove(care_instr_excel_org)
                        care_box = True 

                for excare_instr_excel in excare_instr_langwise: 
                    excare_instr_org = excare_instr_excel
                    excare_instr_excel = unidecode(excare_instr_excel.replace(' ', ''))
                    found_excare, found_excare_idx, excare_dist = find_str_in_gt(excare_instr_excel, pdf_text, return_indices=True)
                    
                    if found_excare: 
                        found_excare_str = pdf_text[found_excare_idx:found_excare_idx+len(excare_instr_excel)] 
                        user_logger.info(f'MATCHED: {excare_instr_excel} (EXCEL) == {found_excare_str} (PDF)')
                        dev_logger.info(f'MATCHED: {excare_instr_excel} (EXCEL) == {found_excare_str} (PDF)')
                        
                        excare_gt_leftovers[lang].remove(excare_instr_org)
                        excare_box = True 

                if care_box: 
                    care_idxs[lang] = text_idx
                if excare_box: #excare_box and not care_box:
                    excare_idxs[lang] = text_idx 

                    # else: 
                    #     import pdb; pdb.set_trace()
                        
    care_gt_leftovers   = remove_empty_lists(care_gt_leftovers)
    excare_gt_leftovers = remove_empty_lists(excare_gt_leftovers)
    
    if care_gt_leftovers:
        for lang, defect_list in care_gt_leftovers.items(): 
            for defect in defect_list: 
                if defect == 'EXCLUSIVE OF TRIMMING AND DECORATION' and 'GB' not in lang: 
                    continue
                care_defects['total'] += 1
                if care_idxs[lang] is not None:
                    defect_crop = care_instr[lang][care_idxs[lang]]['lang_crop'] 
                    defect_page_num = care_instr[lang][care_idxs[lang]]['page_num'] 
                else: 
                    defect_page_num = 0
                    defect_crop = np.ones((500,500,3), dtype=np.uint8) * 255

                defect_str = f"DEFECT: Incorrect/Missing Care Instruction: {defect}" + f" in language: {lang} at Page #{defect_page_num}"

                defect_dict = {
                    'defect_str': defect_str, 
                    'defect_crop': defect_crop
                }

                care_defects[lang].append(defect_dict) 
                user_logger.info(defect_str)
                dev_logger.info(defect_str)

    if excare_gt_leftovers:
        for lang, defect_list in excare_gt_leftovers.items(): 
            for defect in defect_list: 
                if defect == 'EXCLUSIVE OF TRIMMING AND DECORATION' and 'GB' in lang: 
                    if exclusive: 
                        continue 
                elif defect == 'EXCLUSIVE OF TRIMMING AND DECORATION' and 'GB' not in lang: 
                    continue 
                
                excare_defects['total'] += 1

                if excare_idxs[lang] is not None:
                    defect_crop = care_instr[lang][excare_idxs[lang]]['lang_crop'] 
                    defect_page_num = care_instr[lang][excare_idxs[lang]]['page_num'] 
                else: 
                    defect_page_num = 0
                    defect_crop = np.ones((500,500,3), dtype=np.uint8) * 255

                
                defect_str = f"DEFECT: Incorrect/Missing Extra Care Instruction: {defect}" + f" in language: {lang} at Page#{defect_page_num}"

                defect_dict = {
                    'defect_str': defect_str, 
                    'defect_crop': defect_crop
                }

                excare_defects[lang].append(defect_dict)
                user_logger.info(defect_str)
                dev_logger.info(defect_str)
    care_defects_refined = remove_duplicate_defects(care_defects)
    excare_defects_refined = remove_duplicate_defects(excare_defects)
    defects['care_instr'] = care_defects_refined 
    defects['excare_instr'] = excare_defects_refined 
    defects['total_defects'] += care_defects_refined['total']
    defects['total_defects'] += excare_defects_refined['total']

    return defects 


def inspect_care_symbols(prototypical_model, prototypes, symbols_found, symbols_gt, defects, threshold=0):
    gt_labels = symbols_gt.keys() 
    num_symbols = len(gt_labels)
    
    defects_symbols = {}
    defects_symbols['total'] = 0

    care_symbols_inspected = False 
    user_logger.info("INFO: Inspecting care symbols")
    dev_logger.info("INFO: Inspecting care symbols")
    matched_record = {}
    
    for page_num, page_list in symbols_found.items():
        matched_record[page_num] = 0
        if len(page_list):
            care_symbols_inspected = True
            dev_logger.info(f"\nINFO: {len(page_list)} instance(s) of care symbols found on Page {page_num}")
            symbol_dict = page_list[0]
            symbol_strip = symbol_dict['crop']

            # Process the symbol strip using contours
            gray = cv2.cvtColor(symbol_strip, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours by X coordinate
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

            # Filter contours by size to ignore noise
            min_contour_area = 500  # Adjust this threshold according to your image
            filtered_contours = [cnt for cnt in sorted_contours if cv2.contourArea(cnt) > min_contour_area]

            for symbol_num, contour in enumerate(filtered_contours):
                x, y, w, h = cv2.boundingRect(contour)

                # Expand the bounding box to capture underlines or nearby features
                expand_ratio = 0.2  # Adjust this to capture more context, smaller to avoid overlap
                expand_w = int(w * 0.02)
                expand_h = int(h * expand_ratio)

                x = max(x - expand_w, 0)
                y = max(y - expand_h, 0)
                w = min(w + 2 * expand_w, symbol_strip.shape[1] - x)
                h = min(h + 2 * expand_h, symbol_strip.shape[0] - y)

                # Crop the region of interest (ROI) from the image
                symbol = symbol_strip[y:y+h, x:x+w]
                # Convert symbol to a PIL image
                symbol_image = Image.fromarray(cv2.cvtColor(symbol, cv2.COLOR_BGR2RGB))
                
                
                # Classify the symbol using the Prototypical Network
                predicted_cls, distances = classify_image(prototypical_model, prototypes, symbol_image, threshold)

                # Handle 'k' suffix in predicted classes
                if predicted_cls.endswith('k'):
                    predicted_cls_base = predicted_cls[:-1]
                elif '.1' in predicted_cls:
                    predicted_cls_base = predicted_cls.split('.1')[0]
                else:
                    predicted_cls_base = predicted_cls

                gt_class = list(gt_labels)[symbol_num]
                label_cls = symbols_gt[gt_class]['symbol_id']
                label_str = symbols_gt[gt_class]['symbol_instr']

                # Handle 'k' suffix in GT labels
                if label_cls.endswith('k'):
                    label_cls_base = label_cls[:-1]
                elif '.1' in label_cls:
                    label_cls_base = label_cls.split('.1')[0]
                else:
                    label_cls_base = label_cls

                if predicted_cls_base == label_cls_base:
                    user_logger.info(f"\nMATCHED: {predicted_cls} (PDF) == {label_cls} (EXCEL) - {gt_class}: {label_str}")
                    dev_logger.info(f"\nMATCHED: {predicted_cls} (PDF) == {label_cls} (EXCEL) - {gt_class}: {label_str}")
                else: 
                    defect_str = f"DEFECT: Could not match {gt_class} symbol {label_cls} (EXCEL) with {predicted_cls} (PDF)"
                    
                    defects_symbols['total'] += 1
                    defects_symbols[gt_class] = {
                        'defect_crop': symbol, 
                        'defect_str' : defect_str
                    }

                    user_logger.info(defect_str)
                    dev_logger.info(defect_str)

    defects['total_defects'] += defects_symbols['total']
    defects['care_symbols'] = defects_symbols

    return defects 

def plot_defect(defect_list, unique_defective_pages=None):

    anchor_y = 65
    font_scale = 1
    text_thickness = 2
    width_add = 20
    line_spacing = 20
    border_width = 5

    plotted_defect_list = []
    defective_pdf_pages = []
        
            # if defect_page_num and len(pdf['bboxes']) == len(pdf['images']):
            #     defective_page_bboxes = [{pdf_page_num: pdf['bboxes'][pdf_page_num][defect_page_num]} for pdf_page_num in range(len(pdf['bboxes']))]
                
            #     for _, defect_dict in enumerate(defective_page_bboxes):
            #         for pdf_page_num, page_bbox_list in defect_dict.items():
            #             for bbox in page_bbox_list:
            #                 pdf_page = pdf['images'][pdf_page_num]
            #                 pdf_page = cv2.rectangle(pdf_page, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            #                 defective_pdf_pages.append(pdf_page)
    if defect_list:
        for defect in defect_list:
            defect_str = defect['defect_str']
            page_found, page_found_idx, page_found_dist = find_str_in_gt('Page', defect_str, return_indices=True) 
            if page_found: 
                defect_page_num = extract_numeric_value(defect_str[page_found_idx:])
                unique_defective_pages.add(defect_page_num)

        for defect in defect_list: 
            defect_crop = defect['defect_crop']
            defect_str = defect['defect_str']
            total_text_height = 0
            font = cv2.FONT_HERSHEY_TRIPLEX  
            defect_str_list = defect_str.split('\n')
            max_width = 0
            for line in defect_str_list:
                (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, text_thickness)
                total_text_height += text_height + line_spacing  # Add spacing between lines
                if text_width > max_width:
                    max_width = text_width 
            defect_crop = np.pad(defect_crop, pad_width=((total_text_height+anchor_y, 0), ((max_width-defect_crop.shape[1]+width_add+border_width)//2, (max_width+width_add+border_width)//2), (0, 0)), mode='constant', constant_values=255)

            y_offset = total_text_height
            defect_crop = cv2.copyMakeBorder(defect_crop, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 255))

            for idx, line in enumerate(defect_str_list):
                text_x = width_add  # Adjust x-offset for text alignment
                text_y = int(anchor_y + (idx * (total_text_height / len(defect_str_list))))
                # Draw text in red
                cv2.putText(defect_crop, line, (text_x, text_y), font, font_scale, (0, 0, 255), text_thickness)

            plotted_defect_list.append(defect_crop)

        return plotted_defect_list#, unique_defective_pages

def plot_symbol_defect(defect_dict):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    text_thickness = 2
    border_width = 5 

    defect_str = defect_dict['defect_str']
    defect_crop = defect_dict['defect_crop']

    if 'Confidence' in defect_str:
        defect_str = defect_str.split(' - ')[0]

    # Calculate text size
    text_size, _ = cv2.getTextSize(defect_str, font, font_scale, text_thickness)
    text_width, text_height = text_size

    # Calculate padding needed for the text at the top
    padding_top = text_height + 10  # 10 pixels of additional padding for spacing

    # Create a new image with the additional padding
    padded_image_height = defect_crop.shape[0] + padding_top
    padded_image_width = max(defect_crop.shape[1], text_width)
    padded_image = np.ones((padded_image_height, padded_image_width, 3), dtype=np.uint8) * 255

    # Calculate horizontal offset to center the defect_crop image
    horizontal_offset = (padded_image_width - defect_crop.shape[1]) // 2

    # Place the original image into the new padded image
    padded_image[padding_top:padding_top + defect_crop.shape[0], horizontal_offset:horizontal_offset + defect_crop.shape[1]] = defect_crop

    # Set text position
    text_x = (padded_image_width - text_width) // 2
    text_y = text_height + 5  # 5 pixels of padding from the top

    # Add text to the new image
    cv2.putText(padded_image, defect_str, (text_x, text_y), font, font_scale, (0, 0, 255), text_thickness)
    padded_image = np.pad(padded_image, pad_width=((100,100),(10,10),(0,0)), mode='constant', constant_values=255)
    padded_image = cv2.copyMakeBorder(padded_image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 255))

    return padded_image

def send_pipeline_fail(defect_str):
    defect_crop = plot_pipeline_defect(defect_str)
    user_logger.info(defect_str)
    dev_logger.info(defect_str)

    if gen_cfg['debug_endpoints']:
        import pdb; pdb.set_trace() 

    response = requests.post(f"{gen_cfg['protocol']}://{gen_cfg['frontend_domain']}:{gen_cfg['frontend_port']}/api/inspection/Preprocessed", 
                    json = {
                        'fiber_content': [],
                        'care_instr': [], 
                        'excare_instr': [], 
                        'symbols': [],
                        'pdf': []
                    }, headers={"X-Api-Key": gen_cfg['api_key'],"Content-Type": "application/json"}, verify=True, timeout=30)#Verify for SSL Certifcaite
    return {
            "inspection_status": 2, # 0 for nondefective, 1 defective, 2 error
            "defect_list": [encode_img(defect_crop)],
            "pdf_image": [encode_img(defect_crop)],
            "logs": encode_str(defect_str),
            'time_taken': str(0.0) +' secs'
        }

def plot_pipeline_defect(defect_str):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    text_thickness = 2 
    border_width = 5 
    # Set text size and create a white background image based on text size
    text_size = cv2.getTextSize(defect_str, font, font_scale, text_thickness)[0]
    width, height = text_size[0] + 20, text_size[1] + 20  # Add padding
    background = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Calculate the position to center the text
    text_x = (background.shape[1] - text_size[0]) // 2
    text_y = (background.shape[0] + text_size[1]) // 2
    
    # Add text to the white background
    cv2.putText(background, defect_str, (text_x, text_y), font, font_scale, (0, 0, 255), text_thickness, lineType=cv2.LINE_AA)

    padded_image = np.pad(background, pad_width=((100,100),(10,10),(0,0)), mode='constant', constant_values=255)
    padded_image = cv2.copyMakeBorder(padded_image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 255))
    return padded_image 

def plot_pipeline_defect2(defect_str):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    text_thickness = 2 
    border_width = 5 
    padding_left = 20  # Left padding to align text uniformly
    
    # Split the defect string into lines based on newline character
    lines = defect_str.split('\n')
    
    # Calculate text size for the first line to get the height
    text_size = cv2.getTextSize(lines[0], font, font_scale, text_thickness)[0]
    line_height = text_size[1] + 10  # Line height with some padding
    
    # Calculate the overall height needed
    total_height = line_height * len(lines)
    max_width = max([cv2.getTextSize(line, font, font_scale, text_thickness)[0][0] for line in lines])
    width = max_width + padding_left * 2  # Add padding on both sides
    height = total_height + 40  # Add padding at the top and bottom
    background = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Set the initial y-coordinate
    y = 20 + text_size[1]

    # Add each line of text to the background, left-aligned
    for line in lines:
        cv2.putText(background, line, (padding_left, y), font, font_scale, (0, 0, 255), text_thickness, lineType=cv2.LINE_AA)
        y += line_height

    # Add padding around the image
    padded_image = np.pad(background, pad_width=((100,100),(10,10),(0,0)), mode='constant', constant_values=255)
    padded_image = cv2.copyMakeBorder(padded_image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 255))
    return padded_image 

def remove_duplicate_images(dict_list):
    unique_dict_list = []

    for idx1, item in enumerate(dict_list):
        for idx2, item2 in enumerate(dict_list): 
            if idx1 == idx2:
                continue 

            image = item['image']
            image2 = item2['image']
            if np.array_equal(image, image2):
                continue
            else: 
                unique_dict_list.append(item)
    return unique_dict_list
        


def handle_defects(defects, pdf=None, path=None):

    care_defects     = defects['care_instr']
    excare_defects   = defects['excare_instr']
    symbol_defects   = defects['care_symbols']
    fiber_defects    = defects['fiber_content']
    size_defects     = defects['sizes']
    pipeline_defects = defects['pipeline']
    total_defects    = defects['total_defects']

    user_logger.info(f"INFO: Total {total_defects} defects were found in the PDF.")
    dev_logger .info(f"INFO: Total {total_defects} defects were found in the PDF.")

    plotted_defects = {key:{} for key in defects.keys() if key!='total'}
    plotted_defects['symbol_defects'] = []
    plotted_defects['pipeline'] = []
    plotted_defects['sizes'] = []
    unique_defective_pages = set()

    for defect_lang, defect_list in fiber_defects.items():
        if defect_lang == 'total':
            continue
        plotted_defect_list = plot_defect(defect_list, unique_defective_pages=unique_defective_pages)
        plotted_defects['fiber_content'][defect_lang] = plotted_defect_list

    for defect_lang, defect_list in care_defects.items():
        if defect_lang == 'total':
            continue
        plotted_defect_list = plot_defect(defect_list, unique_defective_pages=unique_defective_pages)
        plotted_defects['care_instr'][defect_lang] = plotted_defect_list

    for defect_lang, defect_list in excare_defects.items():
        if defect_lang == 'total':
            continue
        plotted_defect_list = plot_defect(defect_list, unique_defective_pages=unique_defective_pages)
        plotted_defects['excare_instr'][defect_lang] = plotted_defect_list

    plotted_pdf = {}
    if size_defects: 
        unique_defective_pages.add(1)
    for pdf_page_num, pdf_dict in pdf.items(): 
        pdf_page = pdf_dict['pdf_page']

        subpage_list = pdf_dict['subpage_dict']
        for subpage in subpage_list:
            if subpage['page_num'] in unique_defective_pages: 
                bbox = subpage['bbox']
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                
                defective_overlay = np.zeros_like(pdf_page, dtype=np.uint8)
                defective_overlay[ymin:ymax, xmin:xmax] = (0, 0, 255)
                
                mask = np.zeros_like(pdf_page[:, :, 0], dtype=np.uint8)
                mask[ymin:ymax, xmin:xmax] = 1

                pdf_page[mask == 1] = cv2.addWeighted(pdf_page, 1, defective_overlay, 0.6, 0)[mask == 1]
                # pdf_page[mask == 1] = cv2.addWeighted(pdf_page, 0.9, defective_overlay, 0.1, 0)[mask == 1]
                plotted_pdf[pdf_page_num] = pdf_page
                
    plotted_pdf_nums = sorted(plotted_pdf.keys())
    plotted_pdf_list = [plotted_pdf[key] for key in plotted_pdf_nums]

    for defect_type, defect_dict in symbol_defects.items():
        if defect_type == 'total':
            continue 
        plotted_symbol_defect = plot_symbol_defect(defect_dict)
        plotted_defects['symbol_defects'].append(plotted_symbol_defect)
    
    for defect_idx, defect_str in enumerate(pipeline_defects):
        plotted_pipeline_defect = plot_pipeline_defect(defect_str)
        plotted_defects['pipeline'].append(plotted_pipeline_defect)

    for defect_idx, defect_dict in enumerate(size_defects):
        plotted_size_defect = plot_size_defect(defect=defect_dict)
        plotted_defects['sizes'].append(plotted_size_defect)

    all_defect_list = []
    for key, lang_dict in plotted_defects.items():
        if key == 'symbol_defects' or key == 'pipeline' or key == 'sizes':
            if lang_dict:
                for img in lang_dict: 
                    all_defect_list.append(img)

        else:
            for img_list in lang_dict.values():
                if img_list:
                    for img in img_list:
                        all_defect_list.append(img)
    
    if dev_cfg['save_defects']:
        os.makedirs(path, exist_ok=True)
        for defect_idx, defect in enumerate(all_defect_list):
            cv2.imwrite(os.path.join(path, f'{defect_idx+1}.png'), defect)

    return all_defect_list, plotted_pdf_list

def handle_search_mode_defects(defect_message, images, path=None):
    plotted_pdf_list = []

    if defect_message:
        for img in images:
            # Check if img is a string (e.g., file path) and load it as an image
            if img is None or not isinstance(img, np.ndarray):
                raise ValueError("Invalid image detected in the input list.")

            # Ensure img has the correct shape and dimensions
            height, width, channels = img.shape
            defective_overlay = np.zeros((height, width, channels), dtype=np.uint8)
            defective_overlay[:] = (0, 0, 255)  # Create a red overlay

            mask = np.ones((height, width), dtype=np.uint8)  # Create a mask with 1s

            img[mask == 1] = cv2.addWeighted(img, 1, defective_overlay, 0.6, 0)[mask == 1]
            plotted_pdf_list.append(img)

    if path:
        os.makedirs(path, exist_ok=True)
        for defect_idx, defect in enumerate(plotted_pdf_list):
            cv2.imwrite(os.path.join(path, f'{defect_idx + 1}.png'), defect)

    return plotted_pdf_list if plotted_pdf_list else images  # Return original images if no defects






def create_defect_image(defect_dict, output_dir='output_defect_images'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory is set to:", output_dir)
    
    total_defects = defect_dict.get('total', 0)  # Get the total number of defects

    # Loop through each defect type in the defect dictionary, excluding 'total'
    for defect_type, defect_list in defect_dict.items():
        if defect_type == 'total':
            continue
        
        print(f"Processing defect type: {defect_type}")
        
        # Loop through each defect dictionary in the defect list
        for idx, defect_info in enumerate(defect_list):
            print(f"Processing defect {idx + 1}/{len(defect_list)}")
            
            defect_str = defect_info['defect_str']
            defect_crop = defect_info['defect_crop']
           
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 4
            text_size, _ = cv2.getTextSize(defect_str, font, font_scale, font_thickness)
            max_crop_width = defect_crop.shape[1]
            max_text_width = text_size[0] + 20  # Add padding
            
            # Determine the final width of the image
            final_width = max(max_crop_width, max_text_width)
            
            # Create a blank image for the text part
            text_height = text_size[1] + 10  # Adjust according to the font size
            space_height = 10  # Adjust according to the space needed between text and crop
            text_image_height = text_height * 2 + space_height  # Extra space for both texts and spacing
            
            text_image = np.ones((text_image_height, final_width, 3), dtype=np.uint8) * 255  # White background
            
            # Calculate text positions
            text_x = 10
            text_y1 = text_height
            text_y2 = text_y1 + text_height + 10  # Adding 10 for extra spacing between the two texts
            
            # Put the defect string and total defects count on the image
            cv2.putText(text_image, defect_str, (text_x, text_y1), font, font_scale, (0, 0, 255), font_thickness)
            # cv2.putText(text_image, f"Total defects: {total_defects}", (text_x, text_y2), font, font_scale, (0, 0, 255), font_thickness)
            
            # Now combine text image and defect crop
            crop_height, crop_width, _ = defect_crop.shape
            combined_height = text_image_height + crop_height
            combined_image = np.ones((combined_height, final_width, 3), dtype=np.uint8) * 255  # White background
            
            # Place text image at the top
            combined_image[:text_image_height, :final_width] = text_image
            
            # Place defect crop at the bottom
            combined_image[text_image_height:text_image_height + crop_height, :crop_width] = defect_crop
            
            # Add red border line to the image
            combined_image = cv2.copyMakeBorder(combined_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
            
            # Save the image
            output_path = os.path.join(output_dir, f"defect_{defect_type}_{idx + 1}.png")
            cv2.imwrite(output_path, combined_image)
            print(f"Saved defect image: defect_{defect_type}_{idx + 1}.png")


def find_in_slice(slice_text, lang):

    slice_text = slice_text.replace(' ', '').lower()
    lang = lang.lower() + ":"
    if lang in slice_text:
        found_slice, found_slice_idx, slice_dist = find_str_in_gt(lang, slice_text, return_indices=True)
    else: 
        found_slice = False
        found_slice_idx = 0 
        slice_dist = 10000
    return found_slice, found_slice_idx, slice_dist


def slicer(page, stride, slice_y_offset, all_langs, num_slices, paddle):

    halved_crop_h = page.shape[0]

    for slice_idx in range(num_slices):
        lang_in_slice = False
        start = max(0, (slice_idx * stride) - slice_y_offset)
        end = min(((slice_idx + 1) * stride) + slice_y_offset, halved_crop_h)

        current_slice = page[start:end, :]
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()
        current_text = paddle.ocr(current_slice, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()
        if current_text[0] is not None: 
            for det in current_text[0]:
                det_bbox =  det[0]
                det_txt  =  det[1][0]
                for lang in all_langs:
                    found_slice, found_slice_idx, slice_dist = find_in_slice(det_txt, lang)

                    if found_slice and found_slice_idx == 0 and slice_dist < 2: 
                        lang_in_slice = True 
                        pass 

                if not lang_in_slice:
                    halved_slice = current_slice[:, 0:current_slice.shape[1]//2]
                    halving_on = not lang_in_slice or halved_slice.shape[1] < 10
                    while halving_on:
                        torch.cuda.empty_cache()
                        paddle.device.cuda.empty_cache()
                        halved_text = paddle.ocr(halved_slice, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
                        torch.cuda.empty_cache()
                        paddle.device.cuda.empty_cache()
                        if halved_text[0] is not None: 
                            for half_det in halved_text[0]:
                                half_det_bbox =  half_det[0]
                                half_det_txt  =  half_det[1][0]
                                for lang in all_langs:
                                    found_half_slice, found_half_slice_idx, half_slice_dist = find_in_slice(half_det_txt, lang)

                                    if found_half_slice and found_half_slice_idx == 0 and half_slice_dist < 2: 
                                        lang_in_slice = True 
                                    else:
                                        continue 
            if not halving_on:
                continue 

def detect_rectangles(image, min_area=10000, max_area=50000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                rectangles.append(approx)
    # Draw rectangles
    for rect in rectangles:
        cv2.drawContours(image, [rect], -1, (0, 255, 0), 2)

    crops = []
    for rect in rectangles:
        x, y, w, h = cv2.boundingRect(rect)
        crop = image[y:y+h, x:x+w]
        crops.append(crop)

    return image, crops
def plot_boxes(image_path, detections):
    # Read the image
    image = image_path#cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        raise ValueError("Image not found or unable to load")
    
    for detection in detections[0]:
        box, (text, confidence) = detection
        
        # Extract the coordinates of the box
        pt1 = (int(box[0][0]), int(box[0][1]))
        pt2 = (int(box[1][0]), int(box[1][1]))
        pt3 = (int(box[2][0]), int(box[2][1]))
        pt4 = (int(box[3][0]), int(box[3][1]))
        
        # Draw the bounding box
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        cv2.line(image, pt2, pt3, (0, 255, 0), 2)
        cv2.line(image, pt3, pt4, (0, 255, 0), 2)
        cv2.line(image, pt4, pt1, (0, 255, 0), 2)
    return image 
    

def search_mode_function(excel_io, sheet_names, column_names, item_pngs, results_folder, dev_logger, user_logger, vrd_unbatched, vrd_en, encode_img, encode_str, user_log_name, ts_pipeline, gt, item, singlekeyword):    
    search_results = defaultdict(list)
        # Load the specified sheet and handle row and column extraction
        # Initialize an empty dictionary for unique values, one for each column name
    unique_values = {col: set() for col in column_names}

    # Iterate through each sheet in the list of sheet names
    for sheet in sheet_names:
        print(sheet)
        try:
            # Read the sheet without headers
            df = pd.read_excel(excel_io, sheet_name=sheet, header=None)

            # Dynamically find the header row by searching for the first row that contains all required columns
            header_row = None
            for i, row in df.iterrows():
                # Clean the row values to strip spaces and newlines for comparison
                cleaned_row = row.str.strip().str.replace('\n', '')
                if any(col in cleaned_row.values for col in column_names):
                    header_row = i
                    break

            if header_row is None:
                continue  # Skip this sheet if the header row is not found

            # Set column headers based on the identified header row
            df.columns = df.iloc[header_row]

            # Drop all rows up to and including the header row
            df = df.iloc[header_row + 1:]

            # Strip spaces and newlines from column names
            df.columns = df.columns.str.strip().str.replace('\n', '')

            # Iterate over each required column and collect unique values
            for col in column_names:
                if col in df.columns:
                    unique_values[col].update(df[col].dropna().unique().tolist())

        except Exception as e:
            dev_logger.error(f"Error processing sheet '{sheet}': {e}")
            continue

    # Convert sets to lists for final output
    unique_values = {col: list(values) for col, values in unique_values.items()}

    # Log the unique values found
    for col, values in unique_values.items():
        dev_logger.info(f"Unique values for {col}: {values}")

    # Handle case where no unique values were found
    if all(len(values) == 0 for values in unique_values.values()):
        defect_str = "No unique values found in the specified columns across all sheets."
        dev_logger.error(defect_str)
        user_logger.error(defect_str)
        return send_pipeline_fail(defect_str)
    gt['fc_gt'] = unique_values
    frontend_gt = send_preprocessed_inputs(gt.copy(),search_mode=True)
    pdf_image = item_pngs[item]['img_list']

    # OCR processing across all pages and items

    # Initialize dictionaries to store search results, matched words, and unmatched words
    search_results = defaultdict(list)
    matched_words = []
    unmatched_keywords = []
    matched_words2 = []
    unmatched_words2 = []


    slice_params = {
        'horizontal_stride': 250,  # Moderate stride for better overlap
        'vertical_stride': 150,    # Moderate stride for better overlap
        'merge_x_thres': 25,       # Moderate threshold for horizontal merging
        'merge_y_thres': 20        # Moderate threshold for vertical merging
    }

    # OCR processing across all pages and items
    for item_name, item_pages in item_pngs.items():
        dev_logger.info(f"{item_name} contains {len(item_pages['img_list'])} pages.")
        all_page_text = ""

        pdf_images_with_highlights = []
        # Iterate through each page of the item
        for pdf_page_num, item_page in enumerate(item_pages["img_list"]):
            highlighted_image = item_page.copy()  # Create a copy for drawing rectangles

            # Perform OCR on the entire page
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            ocr_results = vrdocr_2.ocr(item_page, det=True, rec=True, cls=False, slice=slice_params, bin=False, inv=False, alpha_color=(255, 255, 255))
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            # Extract and concatenate text from OCR results
            page_text = ""
            if ocr_results:
                for res in ocr_results:
                    if res:
                        for elem in res:
                            word = elem[1][0].lower()
                            page_text += word + " "

                            # Check if the word is in the unique values and draw a rectangle if matched
                            if any(word == str(value).lower() for values in unique_values.values() for value in values):
                                x_min, y_min, x_max, y_max = int(elem[0][0][0]), int(elem[0][0][1]), int(elem[0][2][0]), int(elem[0][2][1])
                                cv2.rectangle(highlighted_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                matched_words2.append(word)
                            else:
                                unmatched_words2.append(word)

            all_page_text += page_text
            pdf_images_with_highlights.append(highlighted_image)
            # Detect rectangles and get cropped images
            detected_image, crops = detect_rectangles(item_page, min_area=5000, max_area=3000000)
            # Process each cropped image
            for crop in crops:
                # Perform OCR on the flipped image
                torch.cuda.empty_cache()
                paddle.device.cuda.empty_cache()
                ocr_results_flipped = vrd_en.ocr(crop, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
                torch.cuda.empty_cache()
                paddle.device.cuda.empty_cache()
                # Extract and concatenate text from OCR results of flipped image
                flipped_text = ""
                if ocr_results_flipped:
                    for res in ocr_results_flipped:
                        if res:
                            flipped_text += ' '.join([elem[1][0] for elem in res if elem[1][0]]) + " "

                # Append flipped text to all_page_text
                all_page_text += flipped_text

            detected_image, crops = detect_rectangles(item_page, min_area=70000, max_area=300000)
            # Process each cropped image
            for crop in crops:
                # Flip the image
                flipped_image = cv2.flip(crop, 0)
                flipped_image = cv2.flip(flipped_image, 1)
                # Perform OCR on the flipped image
                ocr_results_flipped = vrd_en.ocr(flipped_image, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)

                # Extract and concatenate text from OCR results of flipped image
                flipped_text = ""
                if ocr_results_flipped:
                    for res in ocr_results_flipped:
                        if res:
                            flipped_text += ' '.join([elem[1][0] for elem in res if elem[1][0]]) + " "

                # Append flipped text to all_page_text
                all_page_text += flipped_text

        # Convert all_page_text to lowercase for case-insensitive matching
        all_page_text_lower = all_page_text.lower()


            # Check if single keywords are present in the OCR text
        for keyword in singlekeyword:
            keyword_lower = keyword.lower()
            if keyword_lower not in all_page_text_lower:
                unmatched_keywords.append(keyword)
        # Dictionary to store unmatched words by column
        unmatched_words_dict = {col: [] for col in unique_values.keys()}

        # Match unique values (converted to lowercase) with lowercase combined OCR text
        for col, values in unique_values.items():
            for value in values:
                value = str(value)
                value_lower = value.lower()
                if value_lower in all_page_text_lower:
                    # Store matched result
                    search_results[item_name].append({
                        "column": col,
                        "value": value,
                        "text": all_page_text
                    })
                    matched_words.append(value)
                else:
                    # Store unmatched result in the dictionary
                    unmatched_words_dict[col].append(value)

    # Log matched and unmatched words
    dev_logger.info(f"Matched words: {matched_words}")
    dev_logger.info(f"Unmatched words: {unmatched_words_dict}")

    # Do something with search_results if needed
    # Example: store or further process the search results
    if not search_results:
        defect_str = "No matching content found for the search query."
        dev_logger.info(defect_str)
        user_logger.info(defect_str)
        return send_pipeline_fail(defect_str)
    else:
        search_output = {"search_results": search_results}

    # Prepare the inspection dictionary
    defects = {}
    defects['care_instr'] = {}
    defects['excare_instr'] = {}
    defects['care_symbols'] = {}
    defects['fiber_content'] = {}
    defects['pipeline'] = []
    defects['total_defects'] = 0
    defect_list = []
    pdf_image = pdf_images_with_highlights
    # Generate the defect message with unmatched words categorized by columns
    defect_message_parts = []
    for col, words in unmatched_words_dict.items():
        if words:  # Only include columns with unmatched words
            words_str = ", ".join(words)
            defect_message_parts.append(f"{col}: {words_str}")

    if defect_message_parts:
        defect_message = "Unmatched words by column:\n" + "\n".join(defect_message_parts)
        defects['total_defects'] += 1

        # Add unmatched keywords to the defect message if any
        if unmatched_keywords:
            keywords_str = ", ".join(unmatched_keywords)
            defect_message += f"\nUnmatched keywords: {keywords_str}"
    else:
        defect_message = None

    # Send the defect message to the plotting function
    if defect_message:
        defect_crop = plot_pipeline_defect2(defect_message)
        defect_list.append(defect_crop)

    # Handle defects and update PDF images for search mode
    pdf_list = handle_search_mode_defects(defect_message, pdf_image, path=os.path.join(results_folder, 'defects'))

    if defect_list:
        pdf_color = (0, 0, 255)
        defect_list = [encode_img(defect_img) for defect_img in defect_list]
    else:
        pdf_color = (0, 255, 0)

    pdf_image = [encode_img(cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=pdf_color)) for img in pdf_list]

    to_pipeline = time.time()
    inspection_status = 1 if defects['total_defects'] else 0
    if inspection_status == 1:
        inspection_dict = {
            "inspection_status": inspection_status,  # 0 for nondefective, 1 defective, 2 error
            "defect_list": [encode_img(defect_crop)],
            "pdf_image": pdf_image,
            "logs": encode_str(open(user_log_name, 'r').read()),
            "time_taken": str(round(to_pipeline-ts_pipeline, 3)) + ' secs',
            "total_defects":defects['total_defects']
        }
    else:
        inspection_dict = {
            "inspection_status": inspection_status,  # 0 for nondefective, 1 defective, 2 error
            "defect_list": defect_list,
            "pdf_image": pdf_image,
            "logs": encode_str(open(user_log_name, 'r').read()),
            "time_taken": str(round(to_pipeline-ts_pipeline, 3)) + ' secs',
            "total_defects":defects['total_defects']
        }
    return inspection_dict

def inspect_sizes(pages, size_gt, ocr, defects):
    dev_logger.info( 'INFO: Starting size inspection...')
    user_logger.info('INFO: Starting size inspection...')

    if not len(pages) == len(size_gt): 
        dev_logger.info( "WARNING: Number of instances in the PDF do not match the number of sizes in ground truth.")
        user_logger.info("WARNING: Number of instances in the PDF do not match the number of sizes in ground truth.")
    
    size_gt_checked = size_gt.copy()
    pages_checked = pages.copy()
    for page_idx, page in enumerate(pages):
        page_dets = ocr.ocr(page, det=True, rec=True) 
        page_text = return_all_strings(page_dets).replace(' ', '')
        for size in size_gt: 
            size_strs = list(size['size_gt'].values())
            size_strs_checked = size_strs.copy()
            for size_str in size_strs: 
                processed_size = unidecode(size_str.replace(' ', '').replace('\n', ''))
                if processed_size in page_text:
                    size_strs_checked.remove(size_str)

            if size_strs_checked:
                continue
            else:
                size_gt_checked.remove(size)
                pages_checked[page_idx] = None
                dev_logger.info(f"MATCHED: {size_strs} in instance #{page_idx} of Page 1.") 
                user_logger.info(f"MATCHED: {size_strs} in instance #{page_idx} of Page 1.")
                break
    

    if size_gt_checked:
        missing_sizes = []
        pages_checked = [page for page in pages_checked if page is not None]
        page_h, page_w = pages_checked[0].shape[:2]
        defective_pages = np.hstack([cv2.resize(page, (page_w, page_h)) for page in pages_checked])
        for size in size_gt_checked:
            missing_size = list(size['size_gt'].values())
            missing_sizes+= missing_size
        
        defect_str = f"DEFECT: Artwork missing the following sizes: \n{missing_sizes}"
        dev_logger.info(defect_str)
        user_logger.info(defect_str)
        defects['total_defects'] += 1
        defects['sizes'].append({
            'defect_crop':defective_pages,
            'defect_str':defect_str, 
        })
        
    return defects 



def plot_size_defect(defect):
    strings = defect['defect_str'].split('\n')
    second_image = defect['defect_crop']
    font=cv2.FONT_HERSHEY_SIMPLEX
    font_scale=1
    thickness=2
    spacing=20
    text_color=(0, 0, 255)
    bg_color=(255, 255, 255)

    def create_text_image(strings, font, font_scale, thickness, spacing, color):
        max_text_width = max([cv2.getTextSize(text, font, font_scale, thickness)[0][0] for text in strings])
        text_height = cv2.getTextSize(strings[0], font, font_scale, thickness)[0][1]
        total_height = len(strings) * (text_height + spacing) - spacing + 25

        text_image = np.ones((total_height, max_text_width, 3), dtype=np.uint8) * 255

        y = text_height
        for text in strings:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            x = (max_text_width - text_size[0]) // 2
            cv2.putText(text_image, unidecode(text), (x, y), font, font_scale, color, thickness)
            y += text_height + spacing

        return text_image

    def pad_image_to_size(image, target_size, color):
        height, width = image.shape[:2]
        target_height, target_width = target_size

        top = 25 # No padding at the top
        bottom = 25
        left = (target_width - width) // 2
        right = target_width - width - left

        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded_image

    def combine_images(text_image, second_image):
        text_height, text_width = text_image.shape[:2]
        img_height, img_width = second_image.shape[:2]
        if img_height < text_height or img_width < text_width:
            second_image = pad_image_to_size(second_image, (text_height, text_width), bg_color)
        else:
            scale_factor = min(img_width / text_width, img_height / text_height)
            new_width = int(text_width * scale_factor)
            new_height = int(text_height * scale_factor)
            text_image = cv2.resize(text_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            text_image = pad_image_to_size(text_image, (img_height, img_width), bg_color)

        combined_image = np.vstack((text_image, second_image))
        return combined_image

    # Create the text image
    text_image = create_text_image(strings, font, font_scale, thickness, spacing, text_color)

    # Load the second image
    # second_image = cv2.imread(second_image_path)
    # if second_image is None:
    #     raise ValueError("Failed to load the second image.")

    # Combine the images
    combined_image = combine_images(text_image, second_image)
    combined_image = cv2.copyMakeBorder(combined_image, 15, 15, 15, 15, 
                                        cv2.BORDER_CONSTANT, value=(0, 0, 255))

    return combined_image


def production_mode_function(
    item_pngs, results_folder, dev_logger, user_logger, vrd_en,  encode_img, 
    encode_str, user_log_name, ts_pipeline, dev_cfg, fc_check, care_check, excare_check, 
    symbol_check, care_langs_gt, excare_langs_gt, all_langs, blacklisted_langs, fiber_content_gt, 
    care_instr_gt, extra_care_gt, scissors_detector, page_detector, gt, page_bboxes, 
    vrd_latin, fc_langs_gt, prototypical_model, prototypes, care_symbols_gt, singlekeyword,
    size_gt
):
    pages = {i:[] for i in range(20)}
    for item_name, item_pages in item_pngs.items():
        for pdf_page_num, item_page in enumerate(item_pages["img_list"]):
            page_bboxes[pdf_page_num] = {'pdf_page':item_page,'subpage_dict':[]}
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            page_text = vrd_en.ocr(item_page, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            page_elements = [element for sublist in page_text for element in sublist if 'Page' in element[1][0]]
            dev_logger.info(f"INFO: Found {len(page_elements)} instances of keyphrase 'Page' in PDF Page#{pdf_page_num}")
            for sub_page in page_elements: 
                xmin, ymin = sub_page[0][0]
                xmax, ymax = sub_page[0][-2]
                name = sub_page[1][0]
                page_str = f'page_'+str(extract_numeric_value(name))

                if ymax > item_page.shape[0]//2:
                    y_start = int(item_page.shape[0]//2)
                    y_end = item_page.shape[0]
                else: 
                    y_start = 0
                    y_end = item_page.shape[0]//2
                page_bboxes[pdf_page_num]['subpage_dict'].append({
                                    'page_num':extract_numeric_value(name),
                                    'image':item_page,
                                    'bbox':[int(max(0,xmin-dev_cfg["page_w"])), y_start,int(xmax+dev_cfg["page_w"]),y_end]
                                    })
                crop = item_page[y_start:y_end, int(max(0,xmin-dev_cfg["page_w"])):int(xmax+dev_cfg["page_w"])]
                pages[extract_numeric_value(name)].append(crop)

    combined_text = ""
    pages_emptied = remove_empty_lists(pages)
    pages_filtered = {page_num:[] for page_num in pages_emptied.keys()}
    total_pages = len(pages_filtered)
    for page_num, page_list in pages_emptied.items():
        for page_instance, page in enumerate(page_list):
            page_h, page_w = page.shape[:2]
            xmin = page_w//4 
            ymin = page_h//4
            xmax = page_w-page_w//4
            ymax = page_h-page_h//4 

            page_incentered = page[ymin+dev_cfg['incenter_offset']:ymax+dev_cfg['incenter_offset'], xmin:xmax]
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            center_text = vrd_en.ocr(page_incentered, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            page_text = ""
            if center_text:
                for res in center_text:
                    if res:
                        for elem in res:
                            word = elem[1][0].lower()
                            page_text += word + " "
            combined_text+=page_text
    
            if center_text[0] is not None: 
                pages_filtered[page_num].append((page)) 
    pages_filtered = remove_empty_lists(pages_filtered)
    page_h, page_w = get_biggest_page_size(pages_filtered)
    horizontal_stacks = []
    max_pages = max([len(pages) for _, pages in pages_filtered.items()])
    cut_pages_out = cut_pages(scissors_detector, pages_filtered, ocr=vrd_en, item_name=item_name)
    
    defects = {}
    defects['care_instr'] = {}
    defects['excare_instr'] = {}
    defects['care_symbols'] = {}
    defects['fiber_content'] = {}
    defects['sizes'] = []
    defects['pipeline'] = []
    defects['total_defects'] = 0
    defect_list = []
    

    if size_gt and len(pages_filtered[1]):
        defects = inspect_sizes(pages_filtered[1], size_gt=size_gt, ocr=vrd_latin, defects=defects)

    try: 
        pages_filtered, symbols_dict = cut_pages_out[0], cut_pages_out[1]
    except: 
        return cut_pages_out 
    
    for page_num, pages in pages_filtered.items():
        page_count = len(pages)
        if page_count == 1:
            pages_stacked = np.tile(cv2.resize(pages[0], (page_w, page_h)), (max_pages, 1))
        elif page_count == max_pages:
            pages = [cv2.resize(page, (page_w, page_h)) for page in pages]
            pages_stacked = np.hstack(pages)
        else:
            defect_str = f"ERROR: Unsupported PDF configuration: Page {page_num} appeared {page_count} number of times, not divisible by max pages ({max_pages})"
            return send_pipeline_fail(defect_str)
        
        horizontal_stacks.append(pages_stacked)

    hstack_h, hstack_w = get_biggest_stack(horizontal_stacks=horizontal_stacks)
    horizontal_stacks = [cv2.resize(stack, (hstack_w, hstack_h)) for stack in horizontal_stacks]
    entire_tag = np.vstack(horizontal_stacks)
    frontend_gt = send_preprocessed_inputs(gt)  # Assuming 'gt' is defined elsewhere in the full context
    
    pdf_image = item_pngs[item_name]['img_list']  # Original pdf image
    cv2.imwrite(os.path.join(results_folder, f'stack.png'),entire_tag)

    slice_y_offset = dev_cfg['slice_y_offset']
    for idx in range(max_pages):
        # get one tag
        article_tag = entire_tag[:, idx*page_w:(idx+1)*page_w]
        os.makedirs(os.path.join(results_folder, f'articles/'), exist_ok=True)
        cv2.imwrite(os.path.join(results_folder, f'articles/article_{idx+1}.png'),article_tag)

        if item_pages["metadata"]:
                    
            halved_crop = article_tag[:, :article_tag.shape[1]//2]
            halved_crop_h, halved_crop_w = halved_crop.shape[:2]
            
            stride = dev_cfg['slice_stride']
            num_slices = (halved_crop_h + stride - 1) // stride

            # sliced inference
            article_text = {}#{slice_idx: {} for slice_idx in range(num_slices)}
            slice_langs_found = set()
            for slice_idx in range(num_slices):
                start = max(0, (slice_idx * stride) - slice_y_offset)
                end = min(((slice_idx + 1) * stride) + slice_y_offset, halved_crop_h)
                slice = halved_crop[start:end, :]
                slice = np.pad(slice, pad_width=((dev_cfg['pad_pixels'], dev_cfg['pad_pixels']), (0, 0), (0, 0)), mode='constant', constant_values=255)
                torch.cuda.empty_cache()
                paddle.device.cuda.empty_cache()
                slice_text = vrd_en.ocr(slice, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
                torch.cuda.empty_cache()
                paddle.device.cuda.empty_cache()
                page_text = ""
                if slice_text:
                    for res in slice_text:
                        if res:
                            for elem in res:
                                word = elem[1][0].lower()
                                page_text += word + " "
                combined_text+=page_text

                if slice_text[0] is not None: 
                    langs = {}

                    for txt in slice_text[0]:
                        bbox =  txt[0]
                        txt = txt[1][0]
                        for lang in all_langs:
                                search_for = lang.lower() + ":"
                                to_search = txt.lower().replace(' ', '')
                                if search_for in to_search:
                                    found_slice, found_slice_idx, slice_dist = find_str_in_gt(search_for, to_search, return_indices=True)
                                    if found_slice and found_slice_idx == 0:
                                        top_left = bbox[0]
                                        bottom_right = bbox[-2]
                                        new_bbox = [int(val)-dev_cfg['pad_pixels'] for val in top_left+bottom_right]
                                        if lang not in langs.keys():
                                            dev_logger.info(f"\nINFO: Found {lang} slice.")
                                            langs[lang] = new_bbox
                                            slice_langs_found.add(lang)
                        if not langs: 
                            half_slice = slice[:,0:slice.shape[1]//2]
                            # half_slice = half_slice[:, half_slice.shape[1]//2:]
                            torch.cuda.empty_cache()
                            paddle.device.cuda.empty_cache()
                            half_slice_text = vrd_en.ocr(half_slice, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
                            torch.cuda.empty_cache()
                            paddle.device.cuda.empty_cache()
                            page_text = ""
                            if half_slice_text:
                                for res in half_slice_text:
                                    if res:
                                        for elem in res:
                                            word = elem[1][0].lower()
                                            page_text += word + " "
                            combined_text+=page_text

                            if half_slice_text[0] is not None: 
                            
                                for txt in half_slice_text[0]:
                                    bbox =  txt[0]
                                    txt = txt[1][0]
                                    for lang in all_langs:
                                        if lang not in blacklisted_langs['unsupported']:
                                            search_for = lang.lower() + ":"
                                            to_search = txt.lower().replace(' ', '')
                                            if search_for == to_search:
                                                found_half_slice, found_half_slice_idx, half_dist = find_str_in_gt(search_for, to_search, return_indices=True)
                                                if found_half_slice and found_half_slice_idx == 0 < 2:
                                                    top_left = bbox[0]
                                                    bottom_right = bbox[-2]
                                                    new_bbox = [int(val)-dev_cfg['pad_pixels'] for val in top_left+bottom_right]
                                                    if lang not in langs.keys():

                                                        dev_logger.info(f"\nINFO: Found {lang} half-slice. Current langs: {langs}")
                                                        langs[lang] = new_bbox
                                                        slice_langs_found.add(lang)

                    if langs:
                        os.makedirs(os.path.join(results_folder, f'slices/found/'), exist_ok=True)
                        cv2.imwrite(os.path.join(results_folder, f'slices/found/{slice_idx}.png'), slice)
                        article_text[slice_idx] = langs
                    else:
                        os.makedirs(os.path.join(results_folder, f'slices/not_found/'), exist_ok=True)
                        cv2.imwrite(os.path.join(results_folder, f'slices/not_found/{slice_idx}.png'), slice)

                os.makedirs(os.path.join(results_folder, f'slices/all'), exist_ok=True)
                cv2.imwrite(os.path.join(results_folder, f'slices/all/{slice_idx}.png'), slice)
            # shift the lang bboxes to tag scale 
            missing_slice_langs = all_langs.difference(blacklisted_langs['unsupported']).difference(slice_langs_found)
            if missing_slice_langs:
                dev_logger.warning(f"WARNING: Missing language slices for {missing_slice_langs}")
            article_langs = {}
            for slice_idx, slice_langs in article_text.items():
                for slice_lang, slice_lang_bbox in slice_langs.items(): 
                    if slice_lang not in article_langs.keys():
                        article_langs[slice_lang] = []

                    start_y = slice_idx * stride
                    end_y = min((slice_idx + 1) * stride, halved_crop_h)
                    offset_y = start_y

                    x_min_norm = slice_lang_bbox[0] 
                    y_min_norm = slice_lang_bbox[1] + offset_y
                    x_max_norm = slice_lang_bbox[2] 
                    y_max_norm = slice_lang_bbox[3] + offset_y
                    
                    normed_bbox =  [x_min_norm,
                                    y_min_norm,
                                    x_max_norm,
                                    y_max_norm]
                    
                    article_langs[slice_lang].append(normed_bbox)
            # get rid of boxes that are too close to each other (i.e. same language found twice duset_trace to overlap)
            article_h, article_w = article_tag.shape[:2]
            article_langs = remove_close_boxes(article_langs, bbox_dist_thresh=dev_cfg["bbox_remove_thresh"])
            sorted_article_langs = sort_lang_dict(article_langs)
            sorted_article_langs.append([0, article_h, article_w, article_h, 'dummy_lang'])
            fiber_content = {}
            care_instr = {}
            for bbox_idx in range(len(sorted_article_langs)-1):
                current_bbox = sorted_article_langs[bbox_idx]
                next_bbox = sorted_article_langs[bbox_idx+1]

                x1_min, y1_min, x1_max, y1_max, current_lang = current_bbox
                if current_lang not in blacklisted_langs['unsupported']:
                    if bbox_idx == len(sorted_article_langs)-2:
                        prev_bbox_h = bbox_extended[3] - bbox_extended[1]
                        y2_min = current_bbox[1]+ prev_bbox_h
                        y2_max = current_bbox[1]+ 2*prev_bbox_h
                        x2_min = 0 
                        x2_max = article_w 
                    else:
                        x2_min, y2_min, x2_max, y2_max, next_lang = next_bbox
                    
                    new_y_min = min(y1_min, y2_min)
                    new_y_max = max(y1_max, y2_min)
                    
                    bbox_extended = [0, new_y_min-dev_cfg['bbox_ext_offset'], article_w, new_y_max-15]
                    lang_crop = article_tag[bbox_extended[1]:bbox_extended[3], bbox_extended[0]:bbox_extended[2]]
                    lang_crop = np.pad(lang_crop, pad_width=((dev_cfg['pad_pixels'], dev_cfg['pad_pixels']), (0, 0), (0, 0)), mode='constant', constant_values=255)
                    
                    # if current_lang in unique_langs: 
                    #     if current_lang == 'KOR':
                    #         lang_text = PADDLE_korean.ocr(lang_crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                    #     elif current_lang == 'CN':
                    #         lang_text = PADDLE_chinese.ocr(lang_crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                    #     elif current_lang == 'J':
                    #         lang_text = PADDLE_japanese.ocr(lang_crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                    # else: 
                    torch.cuda.empty_cache()
                    paddle.device.cuda.empty_cache()
                    lang_text = vrd_latin.ocr(lang_crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                    torch.cuda.empty_cache()
                    paddle.device.cuda.empty_cache()
                    
                    found_str = return_all_strings(lang_text)
                    save_dict = {
                                'page_num': get_page_number(height = article_h, n = total_pages, k = bbox_extended[1]) + 1,
                                'lang_bbox': bbox_extended, 
                                'lang_crop': lang_crop,
                                'text': found_str,
                                'raw_text': lang_text
                                }
                    if '%' in found_str:
                        if current_lang not in fiber_content.keys():
                            fiber_content[current_lang] = []
                        fiber_content[current_lang].append(save_dict)

                        os.makedirs(os.path.join(results_folder, f'fiber_content/'), exist_ok=True)
                        dev_logger.info(f"INFO: FC text for {current_lang}: {found_str}")
                        cv2.imwrite(os.path.join(results_folder, f'fiber_content/{current_lang}_{bbox_idx}.png'), lang_crop)

                    else: 
                        if current_lang not in care_instr.keys():
                            care_instr[current_lang] = []
                        care_instr[current_lang].append(save_dict) 

                        os.makedirs(os.path.join(results_folder, f'care_instructions/'), exist_ok=True)
                        dev_logger.info(f"INFO: Care text for {current_lang}: {found_str}")
                        cv2.imwrite(os.path.join(results_folder, f'care_instructions/{current_lang}_{bbox_idx}.png'), lang_crop)
                    
                    os.makedirs(os.path.join(results_folder, f'lang_crops/'), exist_ok=True)
                    cv2.imwrite(os.path.join(results_folder, f'lang_crops/{current_lang}_{bbox_idx}.png'), lang_crop)

            if fc_check:
                fc_langs_found = set(fiber_content.keys())
                missing_fc_langs = fc_langs_gt.difference(fc_langs_found)
                extra_fc_langs = fc_langs_found.difference(fc_langs_gt)
                blacklisted_langs['missing'] = blacklisted_langs['missing'] | missing_fc_langs
                blacklisted_langs['extra'] = blacklisted_langs['extra'] | extra_fc_langs 

                if missing_fc_langs:
                    defects['total_defects']+=1
                    defect_str = f"DEFECT: Could not find fiber content in language(s) {missing_fc_langs} in artwork."
                    defects['pipeline'].append(defect_str)
                    user_logger.info(defect_str)
                    dev_logger.info(defect_str)

                elif extra_fc_langs:
                    defects['total_defects']+=1
                    defect_str = f"DEFECT: Found extra/incorrect language(s) for fiber content {missing_fc_langs} in artwork."
                    defects['pipeline'].append(defect_str)
                    user_logger.info(defect_str)
                    dev_logger.info(defect_str)
                else: 
                    user_logger.info(f"MATCHED: All fiber languages are correct.")
                    dev_logger.info(f"MATCHED: All fiber languages are correct.")
            # Inspect fiber content
                defects, exclusive = inspect_fiber_content(fiber_content_gt=fiber_content_gt, 
                                                fiber_content=fiber_content, 
                                                blacklisted_langs=blacklisted_langs, 
                                                defects=defects)
            if care_check or excare_check:
                care_langs_found = set(care_instr.keys())
                care_langs_gt = care_langs_gt | excare_langs_gt
                missing_care_langs = care_langs_gt.difference(care_langs_found)
                extra_care_langs = care_langs_found.difference(care_langs_gt)
                
                blacklisted_langs['missing'] = blacklisted_langs['missing'] | missing_care_langs
                blacklisted_langs['extra'] = blacklisted_langs['extra'] | extra_care_langs 

                if missing_care_langs:
                    defects['total_defects']+=1
                    defect_str = f"DEFECT: Could not find care instruction in language(s) {missing_care_langs} in artwork."
                    defects['pipeline'].append(defect_str)
                    user_logger.info(defect_str)
                    dev_logger.info(defect_str)
                    
                # elif extra_care_langs:
                    # defects['total_defects']+=1
                    # defect_str = f"DEFECT: Found extra/incorrect language(s) for care instruction {extra_care_langs} in artwork."
                    # user_logger.info(defect_str)
                    # dev_logger.info(defect_str)
                    # defects['pipeline'].append(defect_str)
                else:
                    user_logger.info(f"MATCHED: Care instruction languages are correct.")
                    dev_logger.info(f"MATCHED: Care instruction languages are correct.")

                # start care instr inspection 
                if not fc_check:
                    exclusive = True
                defects = inspect_care(care_instr, 
                                    care_instr_gt, 
                                    extra_care_gt, 
                                    blacklisted_langs, 
                                    defects, 
                                    use_decode=True,
                                    exclusive=exclusive)
            if symbol_check:
                defects = inspect_care_symbols(
                                    prototypical_model=prototypical_model,
                                    prototypes=prototypes,
                                    symbols_found=symbols_dict,
                                    symbols_gt=care_symbols_gt,
                                    defects=defects,
                                    threshold=45  # Define your classification threshold here
                                )
            combined_text = combined_text.lower()
            unmatched_keywords = [keyword for keyword in singlekeyword if keyword.lower() not in combined_text]
            if unmatched_keywords:
                defects['unmatched_keywords'] = unmatched_keywords
                defect_str = f"DEFECT: Unmatched keywords found: {', '.join(unmatched_keywords)}"
                defects['pipeline'].append(defect_str)
                user_logger.info(defect_str)
                dev_logger.info(defect_str)

            defect_list, plotted_pdf = handle_defects(defects, pdf=page_bboxes, path=os.path.join(results_folder, 'defects'))
            if defect_list: 
                
                pdf_color = (0, 0, 255)
                defect_list = [encode_img(defect_img) for defect_img in defect_list]
            else:
                pdf_color = (0, 255, 0)
            if plotted_pdf:
                pdf_list = plotted_pdf
            else:
                pdf_list = item_pngs[item_name]['img_list']

            pdf_image = [encode_img(cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=pdf_color)) for img in pdf_list]
            
            if dev_cfg['single_article_mode']:
                break
    to_pipeline = time.time()
    inspection_status =  1 if defects['total_defects'] else 0

    inspection_dict = {
        "inspection_status": inspection_status, # 0 for nondefective, 1 defective, 2 error
        "defect_list": defect_list,
        "pdf_image": pdf_image,
        "logs": encode_str(open(user_log_name, 'r').read()),
        "time_taken": str(round(to_pipeline-ts_pipeline, 3)) +' secs',
        "total_defects":defects['total_defects']
    }

    return inspection_dict 
def layout_mode_function(
    item_pngs, results_folder, dev_logger, user_logger,  vrd_en, encode_img, encode_str, user_log_name, ts_pipeline, 
    dev_cfg, fc_check, care_check, excare_check, symbol_check, care_langs_gt, excare_langs_gt, all_langs, blacklisted_langs, 
    fiber_content_gt, care_instr_gt, extra_care_gt, page_detector, scissors_detector, gt, page_bboxes, vrd_latin, 
    fc_langs_gt, prototypical_model, prototypes, care_symbols_gt, string, item, max_pages,singlekeyword
):
    orig_max_offset = 300
    max_offset = orig_max_offset
    combined_text=""
    for item_name, item_pages in item_pngs.items():
        pdf_pages = {}
        for pdf_page_num, item_page in enumerate(item_pages["img_list"]):
            page_bboxes[pdf_page_num] = {'pdf_page':item_page,'subpage_dict':[]}
            pages = {}
            results = page_detector(item_page, conf=0.6, verbose=False) 
            for result in results: 
                boxes = result.boxes
                
                dev_logger.info(f"INFO: Detected {len(boxes.xyxy.tolist())} pages in {item_name} page num {pdf_page_num+1}")
                for box_idx, symbol_bbox in enumerate(boxes):
                    page_num = None 
                    symbol_bbox = [int(elem) for elem in symbol_bbox.xyxy.tolist()[0]]
                    symbol_crop = item_page[symbol_bbox[1]+5:symbol_bbox[3]-8, symbol_bbox[0]+5:symbol_bbox[2]-5]
                    
                    left,right,top,bot = False, False, False, False 

                    while not left:
                        left_of_page = item_page[symbol_bbox[1]:symbol_bbox[3], symbol_bbox[0]-max_offset:symbol_bbox[0]]
                        if 0 in left_of_page.shape:
                            max_offset -=50
                            continue
                        left = True 

                    max_offset = orig_max_offset
                    while not right:
                        right_of_page = item_page[symbol_bbox[1]:symbol_bbox[3], symbol_bbox[2]:symbol_bbox[2]+max_offset]
                        if 0 in right_of_page.shape:
                            max_offset -=50
                            continue
                        right = True 

                    while not top:
                        top_of_page = item_page[symbol_bbox[1]-max_offset:symbol_bbox[1], symbol_bbox[0]:symbol_bbox[2]]
                        if 0 in top_of_page.shape:
                            max_offset -=50
                            continue
                        top = True 

                    max_offset = orig_max_offset
                    while not bot:
                        bot_of_page = item_page[symbol_bbox[3]:symbol_bbox[3]+max_offset, symbol_bbox[0]:symbol_bbox[2]]
                        if 0 in bot_of_page.shape:
                            max_offset -=50
                            continue
                        bot = True 

                    for crop in [top_of_page, left_of_page, right_of_page, bot_of_page]:
                        torch.cuda.empty_cache()
                        paddle.device.cuda.empty_cache()
                        page_text = vrd_en.ocr(crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                        torch.cuda.empty_cache()
                        paddle.device.cuda.empty_cache()
                        page_text2 = ""
                        if page_text:
                            for res in page_text:
                                if res:
                                    for elem in res:
                                        word = elem[1][0].lower()
                                        page_text2 += word + " "
                        combined_text+=page_text2
                        
                        if page_text[0] is not None:
                            page_elements = [element for sublist in page_text for element in sublist if 'page' in element[1][0].lower().replace(' ', '')]
                            for elem in page_elements: 
                                label = elem[-1][0]
                                numbers = sum(c.isdigit() for c in label)
                                special_chars = sum(c in string.punctuation for c in label)
                                refined_label = label.lower().replace(' ', '')
                                pattern = r'^page([0-9]|10)$'
                                if numbers <= 2 and special_chars == 0 and re.match(pattern, refined_label):
                                    page_num = extract_numeric_value(label)
                    
                    if page_num is None: 
                        page_num = box_idx+1
                    
                    page_bboxes[pdf_page_num]['subpage_dict'].append(
                        {'image':item_page,
                            'page_num':page_num,
                            'bbox':[symbol_bbox[0]+5, symbol_bbox[1]+5,symbol_bbox[2]-5,symbol_bbox[3]-8]})
                    pages[page_num] = symbol_crop

            total_pages = len(pages)
            cut_pages_out = cut_layout_pages(pages, scissors_model=scissors_detector)
            # inspect_sizes([cut_pages_out[0][1]], ) 
            try: 
                cutout_pages, symbols_dict = cut_pages_out[0], cut_pages_out[1]
            except:
                return cut_pages_out 
            
            pdf_pages[pdf_page_num] = cutout_pages 

        if any(list_val > 1 for list_val in [len(val) for key, val in pdf_pages.items()]): 
            pagewise_pages = []
            max_width = 0
            max_height = 0
            pdf_pages_sorted = {key: pdf_pages[key] for key in sorted(pdf_pages)}
            page_dict_lists = {key: [] for sub_dict in pdf_pages_sorted.values() for key in sub_dict}
            counter = 0

            for _, pdf_page_dict in pdf_pages_sorted.items(): 
                pdf_page_dict_sorted = {key: pdf_page_dict[key] for key in sorted(pdf_page_dict)}

                for page_num, page_crop in pdf_page_dict_sorted.items():

                    page_dict_lists[page_num].append(page_crop)
                    max_width = max(max_width, page_crop.shape[1])
                    max_height = max(max_height, page_crop.shape[0])
                    counter +=1 

            entire_tag = []
            page_dict_lists = {key: page_dict_lists[key] for key in sorted(page_dict_lists)}
            for page_list in page_dict_lists.values():
                horizontal_stack = []
                for page in page_list: 
                    page = cv2.resize(page, (max_width, max_height))
                    horizontal_stack.append(page)
                horizontal_stack = np.hstack(horizontal_stack)
                entire_tag.append(horizontal_stack)
            entire_tag = np.vstack(entire_tag)
            page_w = max_width
        else: 
            if len(pdf_pages) == 1:
                if len(list(pdf_pages.values())[0]) == 1:
                    entire_tag = list(list(pdf_pages.values())[0].values())[0]
                    page_w = entire_tag.shape[1]
    frontend_gt = send_preprocessed_inputs(gt)
    defects = {}
    defects['care_instr'] = {}
    defects['excare_instr'] = {}
    defects['care_symbols'] = {}
    defects['fiber_content'] = {}
    defects['pipeline'] = []
    defects['total_defects'] = 0
    defect_list = []
    pdf_image = item_pngs[item]['img_list']  # Original pdf image

    cv2.imwrite(os.path.join(results_folder, f'stack.png'),entire_tag)

    slice_y_offset = dev_cfg['slice_y_offset']
    for idx in range(max_pages):
        # get one tag
        article_tag = entire_tag[:, idx*page_w:(idx+1)*page_w]
        os.makedirs(os.path.join(results_folder, f'articles/'), exist_ok=True)
        cv2.imwrite(os.path.join(results_folder, f'articles/article_{idx+1}.png'),article_tag)

        if item_pages["metadata"]:
                    
            halved_crop = article_tag[:, :article_tag.shape[1]//2]
            halved_crop_h, halved_crop_w = halved_crop.shape[:2]
            
            stride = dev_cfg['slice_stride']
            num_slices = (halved_crop_h + stride - 1) // stride

            # sliced inference
            article_text = {}#{slice_idx: {} for slice_idx in range(num_slices)}
            slice_langs_found = set()
            for slice_idx in range(num_slices):
                start = max(0, (slice_idx * stride) - slice_y_offset)
                end = min(((slice_idx + 1) * stride) + slice_y_offset, halved_crop_h)
                slice = halved_crop[start:end, :]
                slice = np.pad(slice, pad_width=((dev_cfg['pad_pixels'], dev_cfg['pad_pixels']), (0, 0), (0, 0)), mode='constant', constant_values=255)
                torch.cuda.empty_cache()
                paddle.device.cuda.empty_cache()
                slice_text = vrd_en.ocr(slice, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
                torch.cuda.empty_cache()
                paddle.device.cuda.empty_cache()
                page_text = ""
                if slice_text:
                    for res in slice_text:
                        if res:
                            for elem in res:
                                word = elem[1][0].lower()
                                page_text += word + " "
                combined_text+=page_text
                if slice_text[0] is not None: 
                    langs = {}

                    for txt in slice_text[0]:
                        bbox =  txt[0]
                        txt = txt[1][0]
                        for lang in all_langs:
                                search_for = lang.lower() + ":"
                                to_search = txt.lower().replace(' ', '')
                                if search_for in to_search:
                                    found_slice, found_slice_idx, slice_dist = find_str_in_gt(search_for, to_search, return_indices=True)
                                    if found_slice and found_slice_idx == 0:
                                        top_left = bbox[0]
                                        bottom_right = bbox[-2]
                                        new_bbox = [int(val)-dev_cfg['pad_pixels'] for val in top_left+bottom_right]
                                        if lang not in langs.keys():
                                            dev_logger.info(f"\nINFO: Found {lang} slice.")
                                            langs[lang] = new_bbox
                                            slice_langs_found.add(lang)
                        if not langs: 
                            half_slice = slice[:,0:slice.shape[1]//2]
                            # half_slice = half_slice[:, half_slice.shape[1]//2:]
                            torch.cuda.empty_cache()
                            paddle.device.cuda.empty_cache()
                            half_slice_text = vrd_en.ocr(half_slice, det=True, rec=True, cls=False, bin=False, inv=False, alpha_color=False)
                            torch.cuda.empty_cache()
                            paddle.device.cuda.empty_cache()
                            page_text = ""
                            if half_slice_text:
                                for res in half_slice_text:
                                    if res:
                                        for elem in res:
                                            word = elem[1][0].lower()
                                            page_text += word + " "
                            combined_text+=page_text
                            if half_slice_text[0] is not None: 
                            
                                for txt in half_slice_text[0]:
                                    bbox =  txt[0]
                                    txt = txt[1][0]
                                    for lang in all_langs:
                                        if lang not in blacklisted_langs['unsupported']:
                                            search_for = lang.lower() + ":"
                                            to_search = txt.lower().replace(' ', '')
                                            if search_for == to_search:
                                                found_half_slice, found_half_slice_idx, half_dist = find_str_in_gt(search_for, to_search, return_indices=True)
                                                if found_half_slice and found_half_slice_idx == 0 < 2:
                                                    top_left = bbox[0]
                                                    bottom_right = bbox[-2]
                                                    new_bbox = [int(val)-dev_cfg['pad_pixels'] for val in top_left+bottom_right]
                                                    if lang not in langs.keys():

                                                        dev_logger.info(f"\nINFO: Found {lang} half-slice. Current langs: {langs}")
                                                        langs[lang] = new_bbox
                                                        slice_langs_found.add(lang)

                    if langs:
                        os.makedirs(os.path.join(results_folder, f'slices/found/'), exist_ok=True)
                        cv2.imwrite(os.path.join(results_folder, f'slices/found/{slice_idx}.png'), slice)
                        article_text[slice_idx] = langs
                    else:
                        os.makedirs(os.path.join(results_folder, f'slices/not_found/'), exist_ok=True)
                        cv2.imwrite(os.path.join(results_folder, f'slices/not_found/{slice_idx}.png'), slice)

                os.makedirs(os.path.join(results_folder, f'slices/all'), exist_ok=True)
                cv2.imwrite(os.path.join(results_folder, f'slices/all/{slice_idx}.png'), slice)
            # shift the lang bboxes to tag scale 
            missing_slice_langs = all_langs.difference(blacklisted_langs['unsupported']).difference(slice_langs_found)
            if missing_slice_langs:
                dev_logger.warning(f"WARNING: Missing language slices for {missing_slice_langs}")
            article_langs = {}
            for slice_idx, slice_langs in article_text.items():
                for slice_lang, slice_lang_bbox in slice_langs.items(): 
                    if slice_lang not in article_langs.keys():
                        article_langs[slice_lang] = []

                    start_y = slice_idx * stride
                    end_y = min((slice_idx + 1) * stride, halved_crop_h)
                    offset_y = start_y

                    x_min_norm = slice_lang_bbox[0] 
                    y_min_norm = slice_lang_bbox[1] + offset_y
                    x_max_norm = slice_lang_bbox[2] 
                    y_max_norm = slice_lang_bbox[3] + offset_y
                    
                    normed_bbox =  [x_min_norm,
                                    y_min_norm,
                                    x_max_norm,
                                    y_max_norm]
                    
                    article_langs[slice_lang].append(normed_bbox)
            # get rid of boxes that are too close to each other (i.e. same language found twice duset_trace to overlap)
            article_h, article_w = article_tag.shape[:2]
            article_langs = remove_close_boxes(article_langs, bbox_dist_thresh=dev_cfg["bbox_remove_thresh"])
            sorted_article_langs = sort_lang_dict(article_langs)
            sorted_article_langs.append([0, article_h, article_w, article_h, 'dummy_lang'])
            fiber_content = {}
            care_instr = {}
            for bbox_idx in range(len(sorted_article_langs)-1):
                current_bbox = sorted_article_langs[bbox_idx]
                next_bbox = sorted_article_langs[bbox_idx+1]

                x1_min, y1_min, x1_max, y1_max, current_lang = current_bbox
                if current_lang not in blacklisted_langs['unsupported']:
                    if bbox_idx == len(sorted_article_langs)-2:
                        prev_bbox_h = bbox_extended[3] - bbox_extended[1]
                        y2_min = current_bbox[1]+ prev_bbox_h
                        y2_max = current_bbox[1]+ 2*prev_bbox_h
                        x2_min = 0 
                        x2_max = article_w 
                    else:
                        x2_min, y2_min, x2_max, y2_max, next_lang = next_bbox
                    
                    new_y_min = min(y1_min, y2_min)
                    new_y_max = max(y1_max, y2_min)
                    
                    bbox_extended = [0, new_y_min-dev_cfg['bbox_ext_offset'], article_w, new_y_max-15]
                    lang_crop = article_tag[bbox_extended[1]:bbox_extended[3], bbox_extended[0]:bbox_extended[2]]
                    lang_crop = np.pad(lang_crop, pad_width=((dev_cfg['pad_pixels'], dev_cfg['pad_pixels']), (0, 0), (0, 0)), mode='constant', constant_values=255)
                    
                    # if current_lang in unique_langs: 
                    #     if current_lang == 'KOR':
                    #         lang_text = PADDLE_korean.ocr(lang_crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                    #     elif current_lang == 'CN':
                    #         lang_text = PADDLE_chinese.ocr(lang_crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                    #     elif current_lang == 'J':
                    #         lang_text = PADDLE_japanese.ocr(lang_crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                    # else: 
                    torch.cuda.empty_cache()
                    paddle.device.cuda.empty_cache()
                    lang_text = vrd_latin.ocr(lang_crop, det=True,rec=True,cls=False,bin=False,inv=False,alpha_color=False)
                    torch.cuda.empty_cache()
                    paddle.device.cuda.empty_cache()
                    
                    found_str = return_all_strings(lang_text)
                    save_dict = {
                                'page_num': get_page_number(height = article_h, n = total_pages, k = bbox_extended[1]) + 1,
                                'lang_bbox': bbox_extended, 
                                'lang_crop': lang_crop,
                                'text': found_str,
                                'raw_text': lang_text
                                }
                    if '%' in found_str:
                        if current_lang not in fiber_content.keys():
                            fiber_content[current_lang] = []
                        fiber_content[current_lang].append(save_dict)

                        os.makedirs(os.path.join(results_folder, f'fiber_content/'), exist_ok=True)
                        dev_logger.info(f"INFO: FC text for {current_lang}: {found_str}")
                        cv2.imwrite(os.path.join(results_folder, f'fiber_content/{current_lang}_{bbox_idx}.png'), lang_crop)

                    else: 
                        if current_lang not in care_instr.keys():
                            care_instr[current_lang] = []
                        care_instr[current_lang].append(save_dict) 

                        os.makedirs(os.path.join(results_folder, f'care_instructions/'), exist_ok=True)
                        dev_logger.info(f"INFO: Care text for {current_lang}: {found_str}")
                        cv2.imwrite(os.path.join(results_folder, f'care_instructions/{current_lang}_{bbox_idx}.png'), lang_crop)
                    
                    os.makedirs(os.path.join(results_folder, f'lang_crops/'), exist_ok=True)
                    cv2.imwrite(os.path.join(results_folder, f'lang_crops/{current_lang}_{bbox_idx}.png'), lang_crop)

            defects                   = {}
            defects['care_instr']     = {}
            defects['excare_instr']   = {}   
            defects['care_symbols']   = {}
            defects['fiber_content']  = {}
            defects['pipeline']       = []
            defects['total_defects']  = 0

            if fc_check:
                fc_langs_found = set(fiber_content.keys())
                missing_fc_langs = fc_langs_gt.difference(fc_langs_found)
                extra_fc_langs = fc_langs_found.difference(fc_langs_gt)
                blacklisted_langs['missing'] = blacklisted_langs['missing'] | missing_fc_langs
                blacklisted_langs['extra'] = blacklisted_langs['extra'] | extra_fc_langs 

                if missing_fc_langs:
                    defects['total_defects']+=1
                    defect_str = f"DEFECT: Could not find fiber content in language(s) {missing_fc_langs} in artwork."
                    defects['pipeline'].append(defect_str)
                    user_logger.info(defect_str)
                    dev_logger.info(defect_str)

                elif extra_fc_langs:
                    defects['total_defects']+=1
                    defect_str = f"DEFECT: Found extra/incorrect language(s) for fiber content {missing_fc_langs} in artwork."
                    defects['pipeline'].append(defect_str)
                    user_logger.info(defect_str)
                    dev_logger.info(defect_str)
                else: 
                    user_logger.info(f"MATCHED: All fiber languages are correct.")
                    dev_logger.info(f"MATCHED: All fiber languages are correct.")
            # Inspect fiber content
                defects, exclusive = inspect_fiber_content(fiber_content_gt=fiber_content_gt, 
                                                fiber_content=fiber_content, 
                                                blacklisted_langs=blacklisted_langs, 
                                                defects=defects)
            if care_check or excare_check:
                care_langs_found = set(care_instr.keys())
                care_langs_gt = care_langs_gt | excare_langs_gt
                missing_care_langs = care_langs_gt.difference(care_langs_found)
                extra_care_langs = care_langs_found.difference(care_langs_gt)
                
                blacklisted_langs['missing'] = blacklisted_langs['missing'] | missing_care_langs
                blacklisted_langs['extra'] = blacklisted_langs['extra'] | extra_care_langs 

                if missing_care_langs:
                    defects['total_defects']+=1
                    defect_str = f"DEFECT: Could not find care instruction in language(s) {missing_care_langs} in artwork."
                    defects['pipeline'].append(defect_str)
                    user_logger.info(defect_str)
                    dev_logger.info(defect_str)
                    
                # elif extra_care_langs:
                    # defects['total_defects']+=1
                    # defect_str = f"DEFECT: Found extra/incorrect language(s) for care instruction {extra_care_langs} in artwork."
                    # user_logger.info(defect_str)
                    # dev_logger.info(defect_str)
                    # defects['pipeline'].append(defect_str)
                else:
                    user_logger.info(f"MATCHED: Care instruction languages are correct.")
                    dev_logger.info(f"MATCHED: Care instruction languages are correct.")

                # start care instr inspection 
                if not fc_check:
                    exclusive = True
                defects = inspect_care(care_instr, 
                                    care_instr_gt, 
                                    extra_care_gt, 
                                    blacklisted_langs, 
                                    defects, 
                                    use_decode=True,
                                    exclusive=exclusive)
            if symbol_check:
                defects = inspect_care_symbols(
                                    prototypical_model=prototypical_model,
                                    prototypes=prototypes,
                                    symbols_found=symbols_dict,
                                    symbols_gt=care_symbols_gt,
                                    defects=defects,
                                    threshold=45  # Define your classification threshold here
                                )
            combined_text = combined_text.lower()
            unmatched_keywords = [keyword for keyword in singlekeyword if keyword.lower() not in combined_text]
            if unmatched_keywords:
                defects['unmatched_keywords'] = unmatched_keywords
                defect_str = f"DEFECT: Unmatched keywords found: {', '.join(unmatched_keywords)}"
                defects['pipeline'].append(defect_str)
                user_logger.info(defect_str)
                dev_logger.info(defect_str)
            defect_list, plotted_pdf = handle_defects(defects, pdf=page_bboxes, path=os.path.join(results_folder, 'defects'))
            if defect_list: 
                
                pdf_color = (0, 0, 255)
                defect_list = [encode_img(defect_img) for defect_img in defect_list]
            else:
                pdf_color = (0, 255, 0)
            if plotted_pdf:
                pdf_list = plotted_pdf
            else:
                pdf_list = item_pngs[item_name]['img_list']

            pdf_image = [encode_img(cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=pdf_color)) for img in pdf_list]
            
            if dev_cfg['single_article_mode']:
                break
    to_pipeline = time.time()
    inspection_status =  1 if defects['total_defects'] else 0

    inspection_dict = {
        "inspection_status": inspection_status, # 0 for nondefective, 1 defective, 2 error
        "defect_list": defect_list,
        "pdf_image": pdf_image,
        "logs": encode_str(open(user_log_name, 'r').read()),
        "time_taken": str(round(to_pipeline-ts_pipeline, 3)) +' secs',
        "total_defects":defects['total_defects']
    }

    return inspection_dict 