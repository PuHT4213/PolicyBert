import torch




if __name__ == "__main__":
    path_A = "models\\bge-base-zh-v1.5\pytorch_model.bin"
    path_B = "models\ZEN_pretrain_base_v0.1.0\ZEN_pretrain_base_v0.1.0\pytorch_model.bin"
    output_path = "models\\bge-base-zh-v1.5-merge\pytorch_model.bin"

    # 加载两个模型的 state_dict
    state_dict_A = torch.load(path_A, map_location="cpu")
    state_dict_B = torch.load(path_B, map_location="cpu")

    # 对A中的key进行处理，如果不以bert.开头，则加上前缀
    new_state_dict_A = {}
    for key in state_dict_A.keys():
        if not key.startswith("bert.") and not key.startswith("cls."):
            new_key = "bert." + key
            new_state_dict_A[new_key] = state_dict_A[key]

        else:
            new_state_dict_A[key] = state_dict_A[key]

    # 更新A的state_dict
    state_dict_A = new_state_dict_A

    # 对B中的key进行处理，如果以bert.cls开头，则去掉前缀bert.
    new_state_dict_B = {}
    for key in state_dict_B.keys():
        if key.startswith("bert.cls"):
            new_key = key.replace("bert.", "")
            new_state_dict_B[new_key] = state_dict_B[key]
        else:
            new_state_dict_B[key] = state_dict_B[key]

    # 更新B的state_dict
    state_dict_B = new_state_dict_B

    # 合并：保留 A 中的 key，B 中的不重复的才添加
    merged_state_dict = state_dict_B.copy()  # 先加载B
    merged_state_dict.update(state_dict_A)  # A中的key会覆盖B中的key

    # 保存合并后的模型
    torch.save(merged_state_dict, output_path)

    print(f"模型合并完成，保存至 {output_path}")