import pandas as pd
import numpy as np
import traceback 
from utils.config import gen_cfg, dev_cfg
from utils.insp_logger import user_logger, dev_logger 
from utils.pipeline_utils import send_pipeline_fail

def dataframe_to_png(dataframe, filename='table.png'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center', cellLoc='center', colLoc='center')
    plt.savefig(filename)
    plt.close(fig)
    
def extra_2_df(sheet, extra_sheet_range, table_name):
    cell_data = list(extra_sheet_range)
    merged_cell_ranges_copy = sheet.merged_cells.ranges.copy()

    # Check for merged cells and merge them
    for merged_cell_range in merged_cell_ranges_copy:
        min_row, min_col, max_row, max_col = merged_cell_range.min_row, merged_cell_range.min_col, merged_cell_range.max_row, merged_cell_range.max_col
        if min_row >= extra_sheet_range[0][0].row and \
                max_row <= extra_sheet_range[-1][-1].row and \
                min_col >= extra_sheet_range[0][0].column and \
                max_col <= extra_sheet_range[-1][-1].column:
            merged_range = f"{sheet.cell(row=min_row, column=min_col).coordinate}:{sheet.cell(row=max_row, column=max_col).coordinate}"
            sheet.unmerge_cells(merged_range)

    data = [[cell.value for cell in row] for row in cell_data]
    df = pd.DataFrame(data)
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    return df

def table_cleanup(sheet, table):
    cell_range = table['main_sheet_range']
    cell_data = list(cell_range)
    merged_cell_ranges_copy = sheet.merged_cells.ranges.copy()

    # Check for merged cells and merge them
    for merged_cell_range in merged_cell_ranges_copy:
        min_row, min_col, max_row, max_col = merged_cell_range.min_row, merged_cell_range.min_col, merged_cell_range.max_row, merged_cell_range.max_col
        if min_row >= cell_range[0][0].row and \
                max_row <= cell_range[-1][-1].row and \
                min_col >= cell_range[0][0].column and \
                max_col <= cell_range[-1][-1].column:
            merged_range = f"{sheet.cell(row=min_row, column=min_col).coordinate}:{sheet.cell(row=max_row, column=max_col).coordinate}"
            sheet.unmerge_cells(merged_range)
    data = [[cell.value for cell in row] for row in cell_data]
    df = pd.DataFrame(data)
    return df 

def excare_instr_table_handler(sheet, table, ref_sheet):
    try:
        ref_sheet = preprocess_gt(ref_sheet)
        df = table_cleanup(sheet=sheet, table=table)

        search_df = df.applymap(lambda x: str(x).lower().replace(' ', ''))
        excare_idxs = search_df.applymap(lambda x: 'extracareinstruction' in str(x))
        excare_idxs = excare_idxs[excare_idxs].stack().index.tolist()

        if len(excare_idxs) == 1: 
            dev_logger.info(f"INFO: Found {len(excare_idxs)} instance of 'extracareinstructions' in extra care table.")
            excare_rows, excare_cols = excare_idxs[0]
            excare_instr_ids = list({elem for elem in df.loc[excare_rows+1:][excare_cols-1] if elem is not None})
            
            if len(excare_instr_ids):
                excare_instrs = {id: ref_sheet.loc[ref_sheet['ID'] == id] for id in excare_instr_ids}
                return True, excare_instrs 
            
            else:
                defect_str = f"ERROR: Could not find an extra care instruction ID column in the given table. Wrong table range/unsupported order configuration."
                return send_pipeline_fail(defect_str)
        else:
            defect_str = f"ERROR: Could not find keyword 'Extra Care Instruction' in the given table. Wrong table range/unsupported order configuration"
            return send_pipeline_fail(defect_str)
        
    except Exception as e:
        defect_str = f"ERROR: Something went wrong while reading extra care instruction table, wrong table ranges/unsupported configuration."
        dev_logger.info( f"{defect_str} {e}{traceback.format_exc()}")
        return send_pipeline_fail(defect_str)

def care_instr_table_handler(sheet, table, ref_sheet):
    try:
        ref_sheet = preprocess_gt(ref_sheet)
        df = table_cleanup(sheet=sheet, table=table)

        search_df = df.applymap(lambda x: str(x).lower().replace(' ', ''))
        excare_idxs = search_df.applymap(lambda x: 'careinstruction' in str(x))
        excare_idxs = excare_idxs[excare_idxs].stack().index.tolist()

        if len(excare_idxs) == 1: 
            dev_logger.info(f"INFO: Found {len(excare_idxs)} instance of 'careinstruction' in extra care table.")
            excare_rows, excare_cols = excare_idxs[0]
            excare_instr_ids = list({elem for elem in df.loc[excare_rows+1:][excare_cols-1] if elem is not None})
            
            if len(excare_instr_ids):
                excare_instrs = {id: ref_sheet.loc[ref_sheet['ID'] == id] for id in excare_instr_ids}
                return True, excare_instrs 
            
            else:
                defect_str =  f"ERROR: Could not find an care instruction ID column in the given table. Wrong table range/unsupported order configuration."
                return send_pipeline_fail(defect_str)
        else:
            defect_str =  f"ERROR: Could not find keyword 'Care Instruction' in the given table. Wrong table range/unsupported order configuration"
            return send_pipeline_fail(defect_str)
    except Exception as e:
        defect_str = f"ERROR: Something went wrong while reading care instruction table, wrong table ranges/unsupported configuration."
        dev_logger.info( f"{defect_str} {e}{traceback.format_exc()}")
        return send_pipeline_fail(defect_str)

import math 

def adidas_size_handler(
        order_form, 
        base_df, 
        size_table,
        size_langs
):
    try:
        base_to_order = {
            'Size Page':'*Size Page',
            'Sourcing Size Description':'*Sourcing Size'
        }
        order_to_base = {value:key for key, value in base_to_order.items()}
        import pdb; pdb.set_trace()

        size_df = table_cleanup(order_form,size_table)
        size_df = size_df.dropna(how='all')
        size_df = size_df.where(pd.notnull(size_df), None)
    except:
        pass

def us_size_handler(
        order_form,
        base_df,
        size_table,
        size_langs
):
    try:    
        base_to_order = {
                'CUS_SIZE':'*Size', 
                'Sourcing Size':'*Sourcing Size', 
                'Tecn_Notation':'*Tech Notation', 
                'Size Run':'*Size Run' 
                }
        order_to_base = {value:key for key, value in base_to_order.items()}

        size_df = table_cleanup(order_form,size_table)
        size_df = size_df.dropna(how='all')
        size_df = size_df.where(pd.notnull(size_df), None)
        found_cols = list(base_to_order.values())

        aligned_size_df = {}
        for col in size_df.columns:
            if size_df[col][0] in base_to_order.values():
                found_cols.remove(size_df[col][0])
                if size_df[col+1][0] is None or math.isnan(size_df[col+1][0]): 
                    aligned_size_df[size_df[col][0]] = size_df[col+1][1:].tolist()

        aligned_size_df = pd.DataFrame(aligned_size_df)
        all_sizes =[]
        criterion = {}

        base_df = base_df.applymap(lambda x: str(x).strip() if pd.notnull(x) else x)
        for row_num, row in aligned_size_df.iterrows():
            sizes = {}

            for test_key, base_key in order_to_base.items():
                criterion[base_key] = int(row[test_key]) if isinstance(row[test_key], float) else row[test_key]
            criterion = {k: str(v).strip() for k, v in criterion.items()}

            filtered_df = base_df[
                            (base_df['CUS_SIZE'] == criterion['CUS_SIZE']) &
                            (base_df['Sourcing Size'] == criterion['Sourcing Size']) &
                            (base_df['Tecn_Notation'] == criterion['Tecn_Notation']) &
                            (base_df['Size Run'] == criterion['Size Run'])
                        ]

            if not filtered_df.empty:
                for size_lang in size_langs:
                    sizes[size_lang] = filtered_df[size_lang].values[0].replace('\n', '')

            all_sizes.append(
                {   'size_gt':sizes, 
                    'size_values':criterion
                }
            )
        return True, all_sizes 
    except Exception as e:
        # defect_str = f'ERROR: Something went wrong while reading the size table. Please make sure your configuration/base file is correct.'
        # dev_logger.info(defect_str)
        # user_logger.info(defect_str)
        # send_pipeline_fail(defect_str)
        
        return False, e


def fc_table_handler(sheet, table):
    try:
        df = table_cleanup(sheet=sheet, table=table)
        fc_df = df.dropna(how='all')
        fc_df = fc_df.dropna(axis=1, how='all')

        # first check if there is article# in the table 
        search_df = df.applymap(lambda x: str(x).lower().replace(' ', ''))
        article_nums = search_df.applymap(lambda x: 'article#' in str(x))
        article_nums = article_nums[article_nums].stack().index.tolist()
        if len(article_nums) == 1:
            article_num_rows, article_num_cols = article_nums[0]
            unique_articles = {elem for elem in df.loc[article_num_rows+1:][article_num_cols] if elem is not None}
            if len(unique_articles):
                article_gt = {}
                for article in unique_articles:
                    article_clean = article.replace(' ', '')
                    article_gt[article_clean] = {}
                    article_map = search_df.applymap(lambda x: article.lower().replace(' ', '') in str(x))
                    article_idxs = article_map[article_map].stack().index.tolist()
                    for article_idx_idx, (article_idx_row, article_idx_col) in enumerate(article_idxs):
                        article_vals = [elem for elem in df.iloc[article_idx_row].tolist()[1:] if elem is not None]
                        
                        
                        for val in article_vals: 
                            if val in gen_cfg['joinable_material_types']:
                                joinable_idx = article_vals.index(val)
                                for join_idx in range(joinable_idx, -1, -1):
                                    if join_idx == joinable_idx:
                                        continue
                                    if isinstance(article_vals[join_idx], str):
                                        cleaned_val = val.replace('(', '').replace(')', '')
                                        article_vals[join_idx] = article_vals[join_idx] + f'///{val} ({article_vals[join_idx+1]}% {cleaned_val})' #+ val
                                        article_vals.remove(val)
                                        article_vals.remove(article_vals[join_idx+1])
                                        break

                        position = article_vals[0]
                        
                        
                        article_vals = article_vals[1:]
                        pairs = [(article_vals[i], article_vals[i + 1]) for i in range(0, len(article_vals), 2)]
                        article_gt[article_clean][position] = {}
                        for pair in pairs:

                        # for val_idx in range(len(article_vals)-1):
                            current_val = pair[0]
                            next_val = pair[1]

                            if isinstance(current_val, int):
                                article_gt[article_clean][position][next_val] = current_val 
                            elif isinstance(current_val, str):
                                article_gt[article_clean][position][current_val] = next_val 
                # see if they're all the same 
                reference = list(article_gt.values())[0]
                all_equal = all(value == reference for value in article_gt.values())
                if all_equal:
                    dev_logger.info(f"INFO: All articles share the same fiber content values, successfully reduced {article_gt} to {reference}.")
                    return True, reference, fc_df
                else:
                    defect_str = f"ERROR: Fiber content ground truth values don't match across each article. Wrong table range/this order configuration is not supported. "
                    dev_logger.info( f"ERROR: Fiber content ground truth values don't match across each article. Wrong table range/this order configuration is not supported. {article_gt}")
                    return send_pipeline_fail(defect_str)
            else:
                defect_str = f"ERROR: Found 'article#' but no valid article numbers were found. Wrong table range/this order configuration is not supported."
            
                dev_logger.info( f"ERROR: Found 'article#' but no valid article numbers were found. Wrong table range/this order configuration is not supported. {unique_articles}")
                return send_pipeline_fail(defect_str)

        elif len(article_nums) > 1:
            defect_str = f"ERROR: Multiple ({len(article_nums)}) instances of key phrase 'article#' found in order form. Wrong table range/this order configuration is not supported."
            user_logger.info(defect_str)
            return send_pipeline_fail(defect_str)
        elif len(article_nums) == 0:
            dev_logger.info(f"WARNING: Could not find 'article#' in the fiber content table, searching for 'partposition'.")
            # find 'Part Position' or any of its variants in the given df 
            search_df = df.applymap(lambda x: str(x).lower().replace(' ', ''))
            part_position_idx_map = search_df.applymap(lambda x: 'partposition' in str(x))
            part_position_idx = part_position_idx_map[part_position_idx_map].stack().index.tolist()
            if len(part_position_idx) == 0: 
                defect_str = f"ERROR: Could not find 'article#' or 'partposition' in fiber content table. Wrong table range/this order configuraion is not supported."
                return send_pipeline_fail(defect_str)
            elif len(part_position_idx) == 1:
                dev_logger.info(f"INFO: Found valid instance of 'partposition' at {part_position_idx}")
                start_row, start_col = part_position_idx[0][0], part_position_idx[0][1]
                num_rows, num_columns = df.shape
                positions = [(df[start_col][start_row+i], start_row+i) for i in range(num_rows-start_row) if df[start_col][start_row+i] is not None][1:]

                if len(positions):
                    # read row for each position
                    position_dict = {} 

                    for position in positions:
                        position_name, position_idx = position
                        position_dict[position_name] = {}
                        position_vals = [val for val in df.loc[position_idx][start_col+1:] if val is not None]

                        pairs = [(position_vals[i], position_vals[i + 1]) for i in range(0, len(position_vals), 2)]

                        for pair in pairs:

                            current = pair[0]
                            next_val = pair[1]
                            # numeric first
                            if isinstance(current, int):
                                position_dict[position_name][next_val] = current 
                            # string first
                            elif isinstance(current, str):
                                position_dict[position_name][current] = next_val 
                    return True, position_dict, fc_df
                else: 
                    defect_str = f"ERROR: Could not find any valid part positions in fiber content table. Wrong table range/this order configuraion is not supported."
                    return send_pipeline_fail(defect_str)
            elif len(part_position_idx) > 1:
                defect_str = f"ERROR: Multiple ({len(part_position_idx)}) instances of key phrase 'partposition' found in order form. Wrong table range/this order configuration is not supported."
                return send_pipeline_fail(defect_str)
    except Exception as e:
        defect_str = f"ERROR: Something went wrong while reading fiber content table, wrong table ranges/unsupported configuration."
        dev_logger.info( f"ERROR: Something went wrong while reading fiber content table, wrong table ranges/unsupported configuration. {e}{traceback.format_exc()}")
        return send_pipeline_fail(defect_str)

def table_2_df(sheet, table, read_only=False):
    cell_range = table['main_sheet_range']
    table_name = table['Name']
    extra_sheet_range = table['extra_sheet_range']
    
    if table_name != 'extra_care':
        if extra_sheet_range:
            extra_care_df = extra_2_df(sheet, extra_sheet_range=extra_sheet_range, table_name=table_name)

        cell_data = list(cell_range)
        merged_cell_ranges_copy = sheet.merged_cells.ranges.copy()

        # Check for merged cells and merge them
        for merged_cell_range in merged_cell_ranges_copy:
            min_row, min_col, max_row, max_col = merged_cell_range.min_row, merged_cell_range.min_col, merged_cell_range.max_row, merged_cell_range.max_col
            if min_row >= cell_range[0][0].row and \
                    max_row <= cell_range[-1][-1].row and \
                    min_col >= cell_range[0][0].column and \
                    max_col <= cell_range[-1][-1].column:
                merged_range = f"{sheet.cell(row=min_row, column=min_col).coordinate}:{sheet.cell(row=max_row, column=max_col).coordinate}"
                sheet.unmerge_cells(merged_range)

        data = [[cell.value for cell in row] for row in cell_data]
        df = pd.DataFrame(data)
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')

        if extra_sheet_range:
            return df, extra_care_df
        else:
            return df
    
    else:
        # import pdb; pdb.set_trace()
        data = [[cell.value for cell in row] for row in cell_range]
        df = pd.DataFrame(data)
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        # df.set_index('ID', inplace=True)

        return df


def table_2_df_order(sheet, table, read_only=False):
    try:
        cell_range = table['main_sheet_range']
        table_name = table['Name']
        extra_sheet_range = table['extra_sheet_range']
        
        if table_name != 'extra_care':
            if extra_sheet_range:
                extra_care_df = extra_2_df(sheet, extra_sheet_range=extra_sheet_range, table_name=table_name)

            cell_data = list(cell_range)
            merged_cell_ranges_copy = sheet.merged_cells.ranges.copy()

            # Check for merged cells and merge them
            for merged_cell_range in merged_cell_ranges_copy:
                min_row, min_col, max_row, max_col = merged_cell_range.min_row, merged_cell_range.min_col, merged_cell_range.max_row, merged_cell_range.max_col
                if min_row >= cell_range[0][0].row and \
                        max_row <= cell_range[-1][-1].row and \
                        min_col >= cell_range[0][0].column and \
                        max_col <= cell_range[-1][-1].column:
                    merged_range = f"{sheet.cell(row=min_row, column=min_col).coordinate}:{sheet.cell(row=max_row, column=max_col).coordinate}"
                    sheet.unmerge_cells(merged_range)

            data = [[cell.value for cell in row] for row in cell_data]
            df = pd.DataFrame(data)
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')

            if extra_sheet_range:
                return df, extra_care_df
            else:
                return df
        
        else:
            # import pdb; pdb.set_trace()
            data = [[cell.value for cell in row] for row in cell_range]
            df = pd.DataFrame(data)
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            # df.set_index('ID', inplace=True)

            return df
    except Exception as e:
        user_logger.info(f"ERROR: Something went wrong while reading care symbol table, wrong table ranges/unsupported configuration. {e}{traceback.format_exc()}")
        dev_logger.info( f"ERROR: Something went wrong while reading care symbol table, wrong table ranges/unsupported configuration. {e}{traceback.format_exc()}")
        return False, {}, None 

def preprocess_care_gt(care_instr, all_care_instr):
    care_instr = preprocess_gt(care_instr)
    all_care_instr = preprocess_gt(all_care_instr)
    import pdb; pdb.set_trace()    
    try: 
        ids = care_instr['*ID'].tolist()
    except:
        ids = excare_instr['ID'].tolist()
    else:
        print("ERROR: Could not find IDs in care symbol table.")

def preprocess_excare_gt(excare_instr, all_excare_instr):
    excare_instr = preprocess_gt(excare_instr)
    all_excare_instr = preprocess_gt(all_excare_instr)

    try: 
        ids = excare_instr['*ID'].tolist()
    except:
        ids = excare_instr['ID'].tolist()
    else:
        if not ids:
            print("ERROR: Could not find keyword 'IDs' or '*IDs' in care symbol table.")

    ids = [id for id in ids if id is not None]

    excare_instr_dict = {}#{id: all_excare_instr.iloc[id] for id in ids}

    for id in ids: 
        
        try:
            excare_instr_dict[id] = all_excare_instr.loc[all_excare_instr['ID'] == id]#excare_instr_dict[id] = all_excare_instr.iloc[id]
        except: 
            if id == 462: # EXCLUSIVE OF TRIMMING AND DECORATION
                excare_instr_dict[id] = {'GB':'EXCLUSIVE OF TRIMMING AND DECORATION'}


    return excare_instr_dict
def preprocess_fc_gt(df, fibre_content_df):

    num_content_types = len(df['Part Position'])
    fc_gt = {}
    for idx in range(num_content_types):

        fc_row = df.iloc[idx].tolist()
        content_type = fc_row[0]

        fc_gt[content_type] = {}
    
        fc_row = [elem for elem in fc_row[1:] if elem is not None]
        if not len(fc_row) % 2:
            num_content_pairs = len(fc_row)//2
            try: 
                for pair_idx in range(0, len(fc_row), num_content_pairs):
                    material_val = fc_row[pair_idx]
                    material =     fc_row[pair_idx+1]
                    fc_gt[content_type][material] = material_val
            except:
                continue
        else:
            print(f"DEFECT: Could not find a content value for one of the materials in excel sheet: {fc_row}")

    return fc_gt, fibre_content_df

def preprocess_gt(df):
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df 

def extra_care_condtions(extra_care_table, extra_care_df):
    extra_care_condtions = []
    for id in extra_care_table['*ID'].tolist():
        if id != None:

            extra_care_condtions.append(extra_care_df.loc[extra_care_df['ID'] == id])

    return extra_care_condtions

def symbol_extract(extra_care_table, extra_care_df):
    extra_care_condtions = {}
    extra_care_df.set_index('Care Code', inplace=True)
    for id in extra_care_table.iloc[0].tolist()[1:]:
        if id != None:
            extra_care_condtions[id] = extra_care_df.loc[id]['AUS care instruction']
    return extra_care_condtions

def searchByArticleNo(table_df, articleNo):
    data = table_df[table_df['Article'].str.strip() == articleNo]
    return data

def searchByPartPosition(table_df, part_name):
    data = table_df[table_df['Part Position'].str.strip() == part_name]
    return data
