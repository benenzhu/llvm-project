set -e
set -x
rm -rf /A/__code/llvm-project/build/tools/mlir/examples/toy/Ch10/
cp -r /A/__code/llvm-project/build/tools/mlir/examples/toy/Ch0 /A/__code/llvm-project/build/tools/mlir/examples/toy/Ch10/
cd build
if [[ "$1" != "--fast" ]]; then
    mold -run cmake ../llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -G Ninja
fi
declare -x PATH="/usr/local/nvm/versions/node/v16.20.2/bin:/root/.vscode-server-insiders/cli/servers/Insiders-336db9ece67f682159078ea1b54212de7636d88a/server/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
mold -run cmake --build . --target clangd toyc-ch1 toyc-ch0  -j150


#check-mlir
