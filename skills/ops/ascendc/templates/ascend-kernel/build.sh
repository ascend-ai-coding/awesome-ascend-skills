#!/bin/bash
set -e

BUILD_KERNELS_MODULE="ON"
DEBUG_MODE="OFF"

while getopts ":a:hd" opt; do
    case ${opt} in
        a )
            BUILD_KERNELS_MODULE="OFF"
            case "$OPTARG" in
                kernels )
                    BUILD_KERNELS_MODULE="ON"
                    ;;
                * )
                    echo "Error: Invalid Value"
                    echo "Allowed value: kernels"
                    exit 1
                    ;;
            esac
            ;;
        d )
            DEBUG_MODE="ON"
            ;;
        h )
            echo "Use './build.sh' build all modules."
            echo "Use './build.sh -a <target>' to build specific parts of the project."
            echo "    <target> can be:"
            echo "    kernels           Only build ascend_kernel."
            exit 1
            ;;
        \? )
            echo "Error: unknown flag: -$OPTARG" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
        : )
            echo "Error: -$OPTARG requires a value" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))


export DEBUG_MODE=$DEBUG_MODE

SOC_VERSION="${1:-Ascend910_9382}"


### Get Current CANN Toolkit Installation Path
_CANN_TOOLKIT_INSTALL_PATH=$(cat /etc/Ascend/ascend_cann_install.info | grep "Toolkit_InstallPath" | awk -F'=' '{print $2}')
source ${_CANN_TOOLKIT_INSTALL_PATH}/set_env.sh
echo -e "\e[1;32mDetected CANN Toolkit Installation Path: ${_CANN_TOOLKIT_INSTALL_PATH}\e[0m"
echo -e "\e[1;33mDouble Checking Environment Variables:\e[0m"
echo -e "\e[1;32mASCEND_HOME_PATH: ${ASCEND_HOME_PATH}\e[0m"
echo -e "\e[1;32mASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}\e[0m"


ASCEND_INCLUDE_DIR=${ASCEND_TOOLKIT_HOME}/$(arch)-linux/include
CURRENT_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$CURRENT_DIR")
VERSION="1.0.0"
OUTPUT_DIR=$CURRENT_DIR/output
mkdir -p $OUTPUT_DIR
echo "outpath: ${OUTPUT_DIR}"

COMPILE_OPTIONS=""

function build_kernels()
{
    CMAKE_DIR=""
    BUILD_DIR="build"

    cd "$CMAKE_DIR" || exit

    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR

    cmake $COMPILE_OPTIONS \
    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
    -DASCEND_HOME_PATH=$ASCEND_HOME_PATH \
    -DASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR \
    -DSOC_VERSION=$SOC_VERSION \
    -DBUILD_KERNELS_MODULE=$BUILD_KERNELS_MODULE \
    -B "$BUILD_DIR" \
    -S .

    cmake --build "$BUILD_DIR" --target install -j 16
    cd -
}

function make_ascend_kernel_package()
{
    cd python/ascend_kernel || exit

    rm -rf "$CURRENT_DIR"/python/ascend_kernel/dist
    cp -v "${CURRENT_DIR}/config.ini" "${CURRENT_DIR}/python/ascend_kernel/ascend_kernel/"
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/python/ascend_kernel/dist/ascend_kernel*.whl ${OUTPUT_DIR}/
    rm -rf "$CURRENT_DIR"/python/ascend_kernel/dist
    cd -
}

function main()
{

    build_kernels
    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel==0.45.1
    fi
    if [[ "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        make_ascend_kernel_package
    fi

}

main
