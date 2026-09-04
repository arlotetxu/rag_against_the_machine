#!/bin/bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;92m'
YELLOW='\033[0;93m'
DEF_COLOR='\033[0;39m'

make clean
echo -e "${YELLOW} Running lint-stric...${DEF_COLOR}"

# # Ejecuta make lint-strict
if ! make lint-strict; then
	echo -e "${RED}❌ Lint-strict failed!! Commit blocked.${DEF_COLOR}"
	exit 1
fi

# echo -e "${GREEN}✅ Lint-strict passed!${DEF_COLOR}"
# echo ""

# echo -e "${YELLOW} Creating the package with uv build...${DEF_COLOR}"
# # Ejecuta uv build
# if ! uv build; then
# 	echo -e "${RED}❌ Build process failed!! Commit blocked.${DEF_COLOR}"
# 	exit 2
# fi

# echo -e "${GREEN}✅ Build successful!${DEF_COLOR}"
echo -e "${GREEN}✅ All checks passed! Commit allowed.${DEF_COLOR}"
exit 0
