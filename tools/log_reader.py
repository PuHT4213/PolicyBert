import tensorflow as tf
from tensorflow.python.summary.event_accumulator import EventAccumulator

# 指定 tfevents 文件路径
file_path = "results/Mar18_08-05-25_n08n06-aagcpu05a5nq-main/events.out.tfevents.1742285125.n08n06-aagcpu05a5nq-main"

# 解析事件文件
event_acc = EventAccumulator(file_path)
event_acc.Reload()

# 获取所有标量数据
for tag in event_acc.Tags()["scalars"]:
    events = event_acc.Scalars(tag)
    print(f"Tag: {tag}")
    for event in events:
        print(f"Step: {event.step}, Value: {event.value}")
