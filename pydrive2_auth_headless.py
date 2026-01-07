from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# This file must exist in the same directory:
# client_secrets.json

gauth = GoogleAuth()

# Use local webserver auth OFF (HPC-safe)
gauth.LocalWebserverAuth = False
gauth.CommandLineAuth()

# Save credentials for future runs
gauth.SaveCredentialsFile("gdrive_creds.json")

drive = GoogleDrive(gauth)

print("Authentication successful.")
print("Saved credentials to gdrive_creds.json")
