from datasets import load_dataset

ds = load_dataset("SALT-NLP/PrivacyLens")

# save to PrivacyLens
ds.save_to_disk("PrivacyLens")