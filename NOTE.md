1. 发现这个代码和 pct_pytorch 很像，因为这个模型是训过的

2. 运行这个项目，一是测试效果，二是基于它看看能不能写自己的东西

3. 问题在于点云对应的图像没找出来，路径不对吧

4. 现在应该干什么？对下一步是最重要的，读dgcnn代码
    - 输入数据是 (batch_size, fea_dim, num_points)，从 get_graph_feature() 而知

5. 现在ModelNet分类的代码搞清楚了，part seg的代码没理清楚
    - script.sh 第2/3条命令是什么意思？

    - part seg 按照文章的意思，是下游任务，是在分类任务完成后微调的？

    - 但是不对啊，有训练 part seg 代码，而train_crosspoint.py实际上没有加载 分割数据集 ShapenetPart 的代码，实际上训不起来分割模型，
        finetune 和 train都应当用 train_partseg.py

        - 按照现在的理解，partseg和 CrossPoint 所提自监督跨模态对比学习，**关系就不大了**，主要是做了分类

    - 【破案】：[参考作者的回答](https://github.com/MohamedAfham/CrossPoint/issues/6)
        - 我上面理解错了，就像作者所说的，train_crosspoint.py 对于 part segementation，是用来做预训练的，提取的是每个点云的特征，这里用256维向量表示
        - 预训练是为了获取点云通用特征表示，此时预训练还是用的 ShapeNetRender 数据集
        - 但是在 part segementation 做微调的时候，就换成了 ShapeNetPart 数据集，从名字上来看它们不一样，实际可能是经过一些变换，大差不差，但毕竟不是一个数据集，加载方式不一样，处理细节不一样，可以认为是在微调

6. 用 train_crosspoint.py 预训练 part seg 分支时：RuntimeError: OrderedDict mutated during iteration
    - 我怀疑是 train_partseg.py 正在工作，加载数据的问题，冲突了

    - 我对代码做了什么？为什么后来训练一直报这个错误
        - 终于搞清楚了，是因为加了 wandb 日志监控的问题，去掉就没事了
        - 随之而来一个问题是，如何在有 wandb 情况下训练模型

7. 实验中观察到的现象
    - cmid_loss 总是大于 imid_loss，也就是说模态内部更好取得一致，而跨模态相对困难，这是好理解的
        - 说明跨模态没训练好
        - 所以设计loss的时候，加大对 cmid_loss 的惩罚力度
        - 可以把这个可视化出来

7. 分布式训练遇到的问题
    1. 感觉没有真正进入内层 train_loader，跳过去了，这是为什么？
        - 打印出来train_loader长度为0，没加载进来数据，这是什么情况
        - 为什么会这样，路径和实验室服务器一模一样啊
        - 【破案】ShapeNetRender除了加载图片，还加载了点云，而我的点云文件夹 ShapeNet 之前没考过来

    2. 输出信息重复多次，只要一个汇总数据就够了
    3. 类似的问题，写日志只要一个进程写汇总数据就行了
    4. 关键问题：如何汇总不同机器的数据，得到想要的loss, accuracy
        - 【重点突破这个问题】
        
        - 目前训练loss通过 averagemeter 看的，这到底是单卡上结果，还是多卡平均的结果
            - 看样子是单卡，因为每个进程会初始化模型
            - 怎么搞到多卡的数据

    5. test_batch_size 应该设大大点，128

    6. SVM 是在CPU上训练的，因为训练和测试特征都用了 feats.detach().cpu().numpy()

    7. 把李老师服务器的docker装上

    8. train_partseg.py 中测试代码重复，搞成一份

    9. train_crosspoint.py，对dgcnn_seg做预训练的时候，出现问题
        1. passing the keyword argument find_unused_parameters=True to torch.nn.parallel.DistributedDataParallel
        2. making sure all forward function outputs participate in calculating loss.
        If you already have done the above two steps, then the distributed data parallel module wasn't able to locate the output tensors in the return 
        - value of your module's forward function. Please include the loss function and the structure of the return value of forward of your module when reporting this issue (e.g. list, dict, iterable).

        - 【解决了】之前DGCNN_seg，在预训练的情况下，返回3个参数，只有中间的inv_feats用在了损失计算，其余两个值没用到，所以 forward() 返回值仅保留inv_feats即可

    10. 搜集每个epoch内部的平均值太耗时，而且意义不大，还是在每个epoch结束之后搞吧
        - 什么情况，一到测试的时候就卡住了，为什么会卡住？
        - 是wandb的问题吗？
        - 看来也得在rank=0使用wandb

8. 单独测试时，想要快速，得把向量保存下来，直接load