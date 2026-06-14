# Repository Configuration for {{OS_NAME}}
# Generated at: {{TIMESTAMP}}
# For Debian/Ubuntu like systems (DEB-based)

deb {{DEBIAN_MAIN_URL}} {{DEBIAN_CODENAME}} main
deb {{DEBIAN_MAIN_URL}} {{DEBIAN_CODENAME}}-updates main
deb {{DEBIAN_SECURITY_URL}} {{DEBIAN_CODENAME}} updates/main
deb {{DOCKER_CE_URL}} {{DEBIAN_CODENAME}} stable
