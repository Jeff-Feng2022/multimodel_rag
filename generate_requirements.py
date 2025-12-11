import pkg_resources
import os

# 获取所有已安装的包
installed_packages = pkg_resources.working_set

# 创建requirements.txt文件
with open('requirements.txt', 'w', encoding='utf-8') as f:
    # 按名称排序并写入每个包的名称和版本
    for package in sorted(installed_packages, key=lambda x: x.key):
        f.write(f"{package.key}=={package.version}\n")
        print(f"{package.key}=={package.version}")

print("\nrequirements.txt 文件已生成！")