import sys

def preprocess_snap_data(input_file_path, output_file_path):
    """
    读取原始 SNAP 边列表文件，创建连续 ID 映射，并输出 C 语言友好的文件。

    输出文件格式:
    - 第一行: <节点总数> <起始用户映射ID> <目标用户映射ID>
    - 剩余行: <映射后源ID> <映射后目标ID>
    """
    
    # 假设起始和目标用户ID已经作为命令行参数传入 main 函数，此处需要修改 main 函数或将其作为参数传入
    # 为了简化，我们只处理图结构，不处理 start/target 用户ID 的映射，
    # 假设 start/target ID 将在 C 语言程序中输入原始值，并在 C 程序中进行映射查找。

    original_to_new_id = {} # 用于存储原始ID到新ID的映射
    new_id_counter = 0      # 连续的新 ID 计数器
    mapped_edges = []       # 存储映射后的边

    print(f"Phase 1: Reading {input_file_path} and creating ID map...")

    try:
        with open(input_file_path, 'r') as infile:
            for line in infile:
                # 跳过注释行 (SNAP 文件通常以 # 或 % 开头)
                if line.startswith('#') or line.startswith('%') or line.strip() == '':
                    continue

                try:
                    # 尝试解析两个 ID
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    
                    # 假设 ID 是整数或可作为字典键的字符串
                    u_orig, v_orig = parts[0], parts[1]

                    # 映射 ID
                    if u_orig not in original_to_new_id:
                        original_to_new_id[u_orig] = new_id_counter
                        new_id_counter += 1
                    u_new = original_to_new_id[u_orig]

                    if v_orig not in original_to_new_id:
                        original_to_new_id[v_orig] = new_id_counter
                        new_id_counter += 1
                    v_new = original_to_new_id[v_orig]

                    mapped_edges.append((u_new, v_new))

                except Exception as e:
                    print(f"Skipping malformed line: {line.strip()} (Error: {e})")
                    continue

        num_vertices = new_id_counter
        print(f"Mapping complete. Total original users: {len(original_to_new_id)}")
        print(f"Total mapped vertices: {num_vertices}")
        print(f"Total edges: {len(mapped_edges)}")
        
        # 将原始 ID 到新 ID 的映射保存到文件，以便 C 程序中查找 start/target ID
        # 或者，直接将 start/target 映射到新 ID，简化 C 程序
        
        return num_vertices, mapped_edges, original_to_new_id

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None


def write_c_friendly_data(output_file_path, num_vertices, mapped_edges):
    """将映射后的数据写入 C 语言可以直接读取的格式。"""
    print(f"Phase 2: Writing mapped data to {output_file_path}...")
    try:
        with open(output_file_path, 'w') as outfile:
            # 写入第一行: 节点总数
            outfile.write(f"{num_vertices}\n")
            
            # 写入边列表
            for u, v in mapped_edges:
                outfile.write(f"{u} {v}\n")
        
        print("Output file successfully created.")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

# 假设主函数从命令行接收输入和输出文件名
if __name__ == "__main__":
    input_file = 'facebook_combined.txt'
    output_file = 'facebook_processed.txt'
    
    # 示例: 打印一个用户映射，方便用户在 C 程序中查询
    # (注意：如果 start/target ID 需要在 C 中查找，映射字典也需要保存)
    
    num_v, edges, id_map = preprocess_snap_data(input_file, output_file)
    
    if num_v is not None:
        write_c_friendly_data(output_file, num_v, edges)
        
        # 打印 ID 映射 (方便用户找到要查询的 start/target 用户的映射 ID)
        print("\n--- Example ID Mapping (Original ID -> Mapped ID) ---")
        # 仅打印前几个和后几个映射
        items = list(id_map.items())
        
        if len(items) > 10:
            for k, v in items[:5]:
                print(f"Original: {k} -> Mapped: {v}")
            print("...")
            for k, v in items[-5:]:
                print(f"Original: {k} -> Mapped: {v}")
        else:
            for k, v in items:
                print(f"Original: {k} -> Mapped: {v}")