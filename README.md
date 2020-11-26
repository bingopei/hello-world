# hello-world
# pytorch笔记
# 2020/11/18
# 使用CrossEntropyLoss损失函数时，train_Y输入的是类别值，不是one-hot编码格式 	

# 使用CrossEntropyLoss损失函数时，loss = criteon(logits, batch_y) #（logits, batch_y）格式为（float32，int64/long）

# ########可视化网路PyTorchViz make_dot
# net = MLP()
# x=torch.randn(size=(1,3000)).requires_grad_(True)
# y=net(x)
# FCArchitecture=make_dot(y,params=dict(list(net.named_parameters())+[('x',x)]))
# FCArchitecture.format='png'
# FCArchitecture.directory='Graphviz'
# FCArchitecture.view()
# ########可视化网络

# ########datasets,TensorDataset,Dataloader批处理数据；
# train_data = Data.TensorDataset(train_X, train_Y)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_worker=3)
# ########如果num_workers的值大于0，要在运行的部分放进__main__()函数里，才不会有错  ！num_worker>0 卡顿？？？？？？？？？

# file = loadmat(file_path) >>type(file)为字典dict格式
# file_keys = file.keys()   >>file_keys (['__header__', '__version__', '__globals__', 'b0'])

# filenames = os.listdir(d_path) # 获得该文件夹下所有文件名

# start = time.time()

# torch.save(net, 'net_model.pth')
# net = torch.load('net_model.pth')  #加class MLP(nn.model)  数据前处理（读取字典，标准化，转torch，gpu等）

# reshape，view，resize 改shape

# nn.MaxPool1d 默认向下取整 49》24

#  labels =['b00','b01','b02','b03','b04','b05','b06','b07','b08','b09','b10','b11']
#  colors =['c', 'b', 'g', 'r', 'm', 'y', 'k', 'darkgoldenrod','lime', 'hotpink','orangered','chocolate']
#  maker 不接受列表？ 可以用for i, marker in zip(range(17), ['.',',', 'o','v','^','<','>','8','s','p','*','+','D','d','x','|','_']):
#  plt.scatter(value[test_Y == i, 0], value[test_Y == i, 1],s=8,c=colors[i],label=labels[i])# false s=20；  “==” 表示判别>> 返回ture or false
#  plt.legend(loc='best',title='12 kinds of fault',markerscale=3) #markerscale 几倍标记大小
