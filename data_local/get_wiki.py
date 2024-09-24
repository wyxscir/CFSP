from datasets import load_dataset
from datasets import load_from_disk


# traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
# testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

# traindata.save_to_disk("./")
# testdata.save_to_disk("./")


# dataset_1 = load_from_disk("./")
# dataset_2 = load_from_disk("./")

# print(dataset_1['train'])
# print(dataset_2)

# data = load_dataset('wikitext', 'wikitext-2-raw-v1')
# data.save_to_disk("./wiki_all")
dataset_1 = load_from_disk("./wiki_all")
print(dataset_1['train'])