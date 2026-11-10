#!/bin/bash
# The setup script of the Brev launchable: paste this file into the launchable's setup script field. Brev runs it
# once, when the instance is created, and marks the instance ready without waiting for it to finish; progress is in
# ~/ss26-local-ai-workshop/setup.log.
# Keep the commands out of `sudo -i ... bash -c '<several lines>'`: sudo -i escapes the newlines, bash then reads the
# lines as one command, and nothing runs.
run='
  cd ~
  [ -d ss26-local-ai-workshop ] || git clone https://github.com/yvbbrjdr/ss26-local-ai-workshop
  cd ss26-local-ai-workshop
  ./setup.sh
'
if [ "$(id -u)" -eq 0 ]; then
  sudo -H -u ubuntu bash -c "$run"
else
  bash -c "$run"
fi
