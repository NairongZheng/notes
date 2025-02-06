- [命令操作](#命令操作)

## 命令操作
1. 打包：`pyinstaller --onefile --noconsole <py_file_path>`
2. proto编译：`cd <proto_file_path>`&&`python -m grpc_tools.protoc -I./ --python_out=./ --grpc_python_out=./ ./<proto_file_name.proto>`（`pip install grpcio-tools==1.47.0`这个版本编译完有类型声明，新版本没有。新版本好像高效一点，但是可读性差一些）
3. 生成代码关系结构：`pip install pylint`&&`sudo apt-get install graphviz`&&`pyreverse -o png -p <pic_name> <folder_or_file_path>`