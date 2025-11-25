import torch
import zipfile

MODEL_ZIP_PATH = "Modelle/model.zip"
EXTRACTED_POLICY_PATH = "Modelle/policy.pth"
OUTPUT_ACTOR_PATH = "Modelle/actor.pth"

# Extract the policy file from the zip archive
print("Extracting the policy from the zip file...")
with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extract("policy.pth", "Modelle")

# Load the policy state dict
print("Loading the policy state dict...")
policy_state_dict = torch.load(EXTRACTED_POLICY_PATH, map_location=torch.device('cpu'))

# Extract only the actor parameters (keys starting with 'actor.')
actor_state_dict = {k[len("actor."):]: v for k, v in policy_state_dict.items() if k.startswith("actor.")}

# Save the actor state dict
print(f"Saving the actor policy to {OUTPUT_ACTOR_PATH}...")
torch.save(actor_state_dict, OUTPUT_ACTOR_PATH)
print("Actor policy saved successfully.")