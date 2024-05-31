import matplotlib.pyplot as plt
import seaborn as sns


model_name = "VGG11_only_3000"
conf_matrix = [
    [0, 1500],
    [0, 1500]
]

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 28})
plt.xlabel('Predicted labels', fontsize="17")
plt.ylabel('True labels', fontsize="17")
plt.title('Confusion Matrix', fontsize="17")
plt.savefig(f"./{model_name}")
plt.close()