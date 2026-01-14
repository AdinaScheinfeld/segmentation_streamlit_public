from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# This file must exist in the same directory:
# client_secrets.json

gauth = GoogleAuth()

# Ensure we get a refresh_token (needed for long-running / repeated jobs)
gauth.settings["client_config_file"] = "client_secrets.json"
gauth.settings["oauth_scope"] = ["https://www.googleapis.com/auth/drive"]
gauth.settings["get_refresh_token"] = True
gauth.settings["oauth_params"] = {
    "access_type": "offline",
    "prompt": "consent",
}

# HPC-safe auth (prints a URL + asks for a verification code)
gauth.CommandLineAuth()

# Save credentials for future runs
gauth.SaveCredentialsFile("gdrive_creds.json")

drive = GoogleDrive(gauth)

print("Authentication successful.")
print("Saved credentials to gdrive_creds.json")
