# 通过Python启动简单网页应用

## 【实验目的】

使用python通过flask运行一个简单的网页

## 【实验目标】

成功部署一个网页应用

## 【实验步骤】

### 1. 拉取实验代码

在右侧命令行通过git拉取实验代码：

```bash
git clone https://github.com/ccchieh/csapp.git
```

克隆下仓库后进入本教程的目录：

```bash
cd csapp/2-1
```

查看Python代码：

```bash
cat main.py
```

可以看到Python文件只有几行简单的代码：

```Python
from flask import Flask

# 实例化一个Flask
app = Flask(__name__)


# 设置一个网页输出的内容
@app.route('/')
def hello_world():
    return 'Hello,World!'


# 只在测试的时候运行
if __name__ == '__main__':
    # 设置测试使用的端口和ip
    app.run("0.0.0.0", 5000)
```

这个python是通过使用Flask框架，创建一个监听在5000端口的网页，该网页应当输出`Hello,World!`

## 安装Flask

下面我们将通过pip安装flask框架：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flask
```


如果看到类似：
```bash
Installing collected packages: flask
Successfully installed flask-1.1.2
```
则表明安装成功

## 启动网站

启动网站只需要简单的运行Python文件即可：

```python
python main.py
```

## 查看网站

只需要点击右侧的界面按钮即可看到输出：
```
Hello,World!
```
如果失败了显示的是：
```
404 page not found
```
