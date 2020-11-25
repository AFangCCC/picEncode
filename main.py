import tkinter as tk
from tkinter import filedialog
from encoder_lib import *


win_size = "800x600"
origin_image = []
origin_code = []
now_file_name = "h"
Huffman_encode_dict = {}


def walkTree_VLR(root_node: HuffmanNode, symbol=''):
    '''
    前序遍历一个哈夫曼树，同时得到每个元素(叶子节点)的编码，保存到全局的Huffman_encode_dict
    :param root_node: 哈夫曼树的根节点
    :param symbol: 用于对哈夫曼树上的节点进行编码，递归的时候用到，为'0'或'1'
    :return: None
    '''
    # 为了不增加变量复制的成本，直接使用一个dict类型的全局变量保存每个元素对应的哈夫曼编码
    global Huffman_encode_dict

    # 判断节点是不是HuffmanNode，因为叶子节点的子节点是None
    if isinstance(root_node, HuffmanNode):
        # 编码操作，改变每个子树的根节点的哈夫曼编码，根据遍历过程是逐渐增加编码长度到完整的
        root_node.symbol += symbol
        # 判断是否走到了叶子节点，叶子节点的key!=None
        if root_node.key is not None:
            # 记录叶子节点的编码到全局的dict中
            Huffman_encode_dict[root_node.key] = root_node.symbol

        # 访问左子树，左子树在此根节点基础上赋值'0'
        walkTree_VLR(root_node.left_child, symbol=root_node.symbol + '0')
        # 访问右子树，右子树在此根节点基础上赋值'1'
        walkTree_VLR(root_node.right_child, symbol=root_node.symbol + '1')
    return


def file_select():
    global now_file_name
    global origin_image
    global origin_code
    if str(entry_name.get()) == '':
        filename = tk.filedialog.askopenfilename()
    else:
        filename = entry_name.get()
    now_file_name = filename
    dataType = str(filename).split('.')[-1]

    if dataType == 'jpg':
        try:
            origin_image = imread(str(filename))
            lb.config(text="读取成功，您选择的文件是：" + filename)
            entry_name__var.set(str(filename))
            imshow("Origin Image", origin_image)
        except Exception:
            lb.config(text="图片读取失败，请检查图片文件地址或者文件名")
            entry_name__var.set(str(filename))
    elif dataType == 'txt':
        try:
            with open(str(filename)) as f:
                origin_code = f.read()
            lb.config(text="读取成功，您选择的文件是：" + filename)
            entry_name__var.set(str(filename))
        except Exception:
            lb.config(text="编码读取失败，请检查编码文件地址或者文件名")
            entry_name__var.set(str(filename))
        print(origin_code)
    else:
        lb.config(text="导入失败 ！请输入 .txt 或 .jpg 文件")
        entry_name__var.set(str(filename))


def svd_encode_decode():
    pic_file_name = "None"
    code_file_name = "None"
    if len(origin_image) == 0:
        out_mess.config(text="未导入文件 ！请输入 .txt 或 .jpg 文件")
        out = ["None", "None", "None", "None",
               ["None", "None", "None"], "None"]
    else:
        try:
            if str(entry_svd.get()) == '':
                out = zip_image_by_svd(origin_image)
            else:
                out = zip_image_by_svd(
                    origin_image, rate=float(str(entry_svd.get())))
            first_name = str(now_file_name).split('.')[0]
            pic_file_name = first_name + "SVD_Decoded_pic.jpg"
            code_file_name = first_name + "SVD_Encoded_code.txt"
            imshow("Decoded Image", out[0])
            imwrite(pic_file_name, out[0])
            # show()
            with open(code_file_name, 'w') as f:
                f.write(str(list(out[-1])))
                f.close()
        except Exception:
            print("hh")
            out_mess.config(text="图像压缩出错，请更换图片 ！")
            out = [
                "None", "None", "None", "None", [
                    "None", "None", "None"], "None"]

    out_mess.config(text="\
奇异值保留率：{}\n\
所用奇异值数量为：{}\n\
原图大小：{}\n\
压缩后用到的矩阵大小：3 X {}\n\
压缩率为：{}\n\n\
SVD编码后的编码保存为{}文件\n\
SVD解码后的图像保存为{}文件".format(out[1], out[2], out[3], out[4], out[5],
                         code_file_name, pic_file_name))


def huff_encode_decode():
    src_img = cvtColor(origin_image, COLOR_BGR2GRAY)
    # 记录原始图像的尺寸，后续还原图像要用到
    src_img_w, src_img_h = src_img.shape
    # 把图像展开成一个行向量
    src_img_ravel = src_img.ravel()

    # {pixel_value:count}，保存原始图像每个像素对应出现的次数，也就是直方图
    hist_dict = {}
    # {pixel_value:code}，在函数中作为全局变量用到了
    global Huffman_encode_dict

    # 得到原始图像的直方图，出现次数为0的元素(像素值)没有加入
    for p in src_img_ravel:
        if p not in hist_dict:
            hist_dict[p] = 1
        else:
            hist_dict[p] += 1

    # 构造哈夫曼树
    huffman_root_node = createTree(hist_dict)
    # 遍历哈夫曼树，并得到每个元素的编码，保存到Huffman_encode_dict
    walkTree_VLR(huffman_root_node)

    # 根据编码字典编码原始图像得到二进制编码数据字符串
    img_encode = encodeImage(src_img_ravel, Huffman_encode_dict)
    # 把二进制编码数据字符串写入到文件中，后缀为bin
    writeBinImage(
        img_encode,
        now_file_name.replace(
            '.jpg',
            'Huffman_Encoded_code.bin'))

    # 读取编码的文件，得到二进制编码数据字符串
    img_read_code = img_encode
    # #readBinImage(
    #     now_file_name.replace(
    #         '.jpg', 'Huffman_Encoded_code.bin'), len(img_encode))
    # 解码二进制编码数据字符串，得到原始图像展开的向量
    # 这是根据哈夫曼树进行解码的方式
    img_src_val_array = decodeHuffman(img_read_code, huffman_root_node)
    # 这是根据编码字典进行解码的方式，更慢一些
    # img_src_val_array = decodeHuffmanByDict(img_read_code, Huffman_encode_dict)

    # 确保解码的数据与原始数据大小一致
    assert len(img_src_val_array) == src_img_w * src_img_h
    # 恢复原始二维图像
    img_decode = reshape(img_src_val_array, [src_img_w, src_img_h])

    # 显示原始图像与编码解码后的图像
    imshow("src_img", src_img)
    imshow("img_decode", img_decode)
    waitKey(0)
    destroyAllWindows()
    imwrite(now_file_name.replace('.jpg', 'Huffman_Dncoded_pic.jpg'),
            img_decode)
    # 计算平均编码长度和编码效率
    total_code_len = 0
    total_code_num = sum(hist_dict.values())
    avg_code_len = 0
    I_entropy = 0
    for key in hist_dict.keys():
        count = hist_dict[key]
        code_len = len(Huffman_encode_dict[key])
        prob = count / total_code_num
        avg_code_len += prob * code_len
        I_entropy += -(prob * log2(prob))
    S_eff = I_entropy / avg_code_len

    first_name = str(now_file_name).split('.')[0]
    pic_file_name = first_name + "Huffman_Dncoded_pic.jpg"
    code_file_name = first_name + "Huffman_Encoded_code.txt"
    out_mess.config(text="平均编码长度为：{:.3f}\n编码效率为：{:.6f}\n\
SVD编码后的编码保存为了{}文件\n\
SVD解码后的图像保存为了{}文件".format(avg_code_len, S_eff, code_file_name, pic_file_name))


window = tk.Tk()
window.title('图像压缩编码/解码')
window.geometry(win_size)

# 使用说明部分
ex = "软件使用说明部分\n\n\
1. 数据输入方式说明。有两种方式，如果在地址栏输入文件地址，然后点击“导入数据”则按照地址直接读取文件;\n\
如果地址栏未输入任何字符，点击“导入数据”可从本地选择数据.\n注 : 只支持 .jpg 文件输入\n\
2. 本软件可对 jpg 图片进行 SVD 压缩编解码和哈夫曼编解码码两种操作\n\n\
作者：方正豪  2020234248  1102047640@qq.com"
ex_text = tk.StringVar()
ex_label = tk.Label(
    window,
    textvariable=ex_text,
    bg='lightGray',
    width=110,
    height=8, justify="left")
ex_text.set(ex)  # \n输入文件地址/从电脑选择文件
ex_label.place(x=10, y=20, anchor='nw')

# 数据输入部分
y_loc_input = 180
x_loc_input = 50
# 提示信息
input_text = tk.StringVar()
input_label = tk.Label(
    window,
    textvariable=input_text,
    bg='lightGray',
    width=20,
    height=1)
input_text.set("文件输入（图像/编码）：")  # \n输入文件地址/从电脑选择文件
input_label.place(x=x_loc_input, y=y_loc_input, anchor='nw')
# 数据地址栏
entry_name__var = tk.StringVar()
entry_name = tk.Entry(window, textvariable=entry_name__var, width=50)
entry_name.place(x=x_loc_input + 150, y=y_loc_input + 1, anchor='nw')
# 导入按钮
button_sample_data = tk.Button(window, text='导入数据', command=file_select)
button_sample_data.place(x=x_loc_input + 520, y=y_loc_input - 3, anchor='nw')
# 数据读取结果
lb = tk.Label(window, text='')
lb.place(x=x_loc_input, y=y_loc_input + 30)

# 编码/解码选项部分
# -------------------- S V D --------------------
y_loc_code = 280
x_loc_code = 50
# 提示信息
svd_text = tk.StringVar()
svd_label = tk.Label(
    window,
    textvariable=svd_text,
    bg='lightGray',
    width=12,
    height=1)
svd_text.set("SVD编码/解码 :")  # \n输入文件地址/从电脑选择文件
svd_label.place(x=x_loc_code, y=y_loc_code, anchor='nw')
# 参数输入栏
entry_svd__var = tk.StringVar()
entry_svd = tk.Entry(window, textvariable=entry_svd__var, width=10)
entry_svd.place(x=x_loc_code + 95, y=y_loc_code + 1, anchor='nw')
# 导入按钮
button_svd = tk.Button(window, text='SVD运行', command=svd_encode_decode)
button_svd.place(x=x_loc_code + 180, y=y_loc_code - 3, anchor='nw')
# 数据读取结果
lba = tk.Label(window, text='')
lba.place(x=x_loc_code, y=y_loc_code + 30)
# -------------------- H u f f m a n --------------------
y_loc_huff = 280
x_loc_huff = 430
# 提示信息
huff_text = tk.StringVar()
huff_label = tk.Label(
    window,
    textvariable=huff_text,
    bg='lightGray',
    width=16,
    height=1)
huff_text.set("Huffman编码/解码 :")  # \n输入文件地址/从电脑选择文件
huff_label.place(x=x_loc_huff, y=y_loc_huff, anchor='nw')
# 导入按钮
button_huff = tk.Button(window, text='Huffman运行', command=huff_encode_decode)
button_huff.place(x=x_loc_huff + 125, y=y_loc_huff - 3, anchor='nw')


# 结果输出部分
y_loc_out = 350
x_loc_out = 50
# 提示信息
out_text = tk.StringVar()
out_label = tk.Label(
    window,
    textvariable=out_text,
    bg='lightGray',
    width=60,
    height=1)
out_text.set("编码/解码结果输出部分 :")  # \n输入文件地址/从电脑选择文件
out_label.place(x=x_loc_out, y=y_loc_out, anchor='nw')
# 结果显示
out_mess = tk.Label(window, text='hh', justify="left")
out_mess.place(x=x_loc_out, y=y_loc_out + 30)

window.mainloop()
