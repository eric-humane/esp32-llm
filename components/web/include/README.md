# WiFi Configuration

## Security Notice
For security reasons, WiFi credentials are stored in a separate header file (`wifi_config.h`) that is not tracked by git.

## Setup Instructions
1. Copy `wifi_config.h.template` to `wifi_config.h`
2. Edit `wifi_config.h` with your WiFi credentials
3. The file is already added to `.gitignore` to prevent accidental commits

## Why This Approach?
This approach prevents WiFi credentials from being exposed in the repository history while keeping the configuration process simple. 