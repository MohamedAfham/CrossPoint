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