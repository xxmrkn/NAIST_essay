# Abstract
We present a conceptually simple, flexible, and general
framework for object instance segmentation. Our approach
efficiently detects objects in an image while simultaneously
generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster
R-CNN by adding a branch for predicting an object mask in
parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small
overhead to Faster R-CNN, running at 5 fps. Moreover,
Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework.
We show top results in all three tracks of the COCO suite of
challenges, including instance segmentation, bounding-box
object detection, and person keypoint detection. Without
tricks, Mask R-CNN outperforms all existing, single-model
entries on every task, including the COCO 2016 challenge
winners. We hope our simple and effective approach will
serve as a solid baseline and help ease future research in
instance-level recognition. Code will be made available.

我々は、概念的にシンプルで柔軟性があり、一般的なフレームワークである
オブジェクトインスタンスのセグメンテーションのためのフレームワークを紹介します。このアプローチは
画像中のオブジェクトを効率的に検出すると同時に
高品質なセグメンテーションマスクを生成します。マスクR-CNNと呼ばれるこの手法は、Faster
R-CNNを拡張し、既存のバウンディングボックス認識と並行して、オブジェクトマスクを予測するブランチを追加しました。
この手法は、Faster R-CNNを拡張し、既存のバウンディングボックス認識と並行して、オブジェクトマスクを予測するブランチを追加したものです。Mask R-CNNの学習は簡単で、Faster R-CNNにわずかなオーバーヘッドを加えるだけです。
Faster R-CNNのオーバーヘッドはわずかで、5 fpsで動作します。さらに。
マスクR-CNNは、同じフレームワークで人間の姿勢を推定することができるなど、他のタスクに簡単に一般化することができます。
COCOチャレンジシリーズの3つのトラックすべてでトップの結果を示しました。
インスタンス・セグメンテーション、バウンディング・ボックス
オブジェクト検出、人物のキーポイント検出などの課題があります。トリックなしで
マスクR-CNNは、COCOを含むすべての課題において、既存の単一モデルの
トリックなしで、Mask R-CNNは既存の単一モデルのエントリをすべてのタスクで上回り、COCO 2016チャレンジの受賞者を含む
の受賞者です。私たちのシンプルで効果的なアプローチが
このシンプルで効果的なアプローチが、確かなベースラインとなり、将来の
インスタンスレベル認識の研究に役立つことを期待しています。コードは公開される予定です。

# §1
The vision community has rapidly improved object detection and semantic segmentation results over a short period of time. In large part, these advances have been driven
by powerful baseline systems, such as the Fast/Faster RCNN [9, 29] and Fully Convolutional Network (FCN) [24]
frameworks for object detection and semantic segmentation, respectively. These methods are conceptually intuitive
and offer flexibility and robustness, together with fast training and inference time. Our goal in this work is to develop a
comparably enabling framework for instance segmentation.
 Instance segmentation is challenging because it requires
the correct detection of all objects in an image while also
precisely segmenting each instance. It therefore combines
elements from the classical computer vision tasks of object detection, where the goal is to classify individual objects and localize each using a bounding box, and semantic segmentation, where the goal is to classify each pixel into a fixed set of categories without differentiating object instances. Given this, one might expect a complex method
is required to achieve good results. However, we show that
a surprisingly simple, flexible, and fast system can surpass
prior state-of-the-art instance segmentation results.
 Our method, called Mask R-CNN, extends Faster R-CNN
[29] by adding a branch for predicting segmentation masks
on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression (Figure 1). The mask branch is a small FCN applied
to each RoI, predicting a segmentation mask in a pixel-topixel manner. Mask R-CNN is simple to implement and
train given the Faster R-CNN framework, which facilitates
a wide range of flexible architecture designs. Additionally,
the mask branch only adds a small computational overhead,
enabling a fast system and rapid experimentation.
 In principle Mask R-CNN is an intuitive extension of
Faster R-CNN, yet constructing the mask branch properly
is critical for good results. Most importantly, Faster R-CNN
was not designed for pixel-to-pixel alignment between network inputs and outputs. This is most evident in how
RoIPool [14, 9], the de facto core operation for attending
to instances, performs coarse spatial quantization for feature extraction. To fix the misalignment, we propose a simple, quantization-free layer, called RoIAlign, that faithfully
preserves exact spatial locations., RoIAlign has a large impact: it improves mask accuracy by relative 10% to 50%, showing
bigger gains under stricter localization metrics. Second, we
found it essential to decouple mask and class prediction: 
we predict a binary mask for each class independently, without
competition among classes, and rely on the network’s RoI classification branch to predict the category. In contrast,
FCNs usually perform per-pixel multi-class categorization,
which couples segmentation and classification, and based
on our experiments works poorly for instance segmentation.
Without bells and whistles, Mask R-CNN surpasses all
previous state-of-the-art single-model results on the COCO
instance segmentation task [23], including the heavilyengineered entries from the 2016 competition winner. As
a by-product, our method also excels on the COCO object
detection task. In ablation experiments, we evaluate multiple basic instantiations, which allows us to demonstrate its
robustness and analyze the effects of core factors.
Our models can run at about 200ms per frame on a GPU,
and training on COCO takes one to two days on a single
8-GPU machine. We believe the fast train and test speeds,
together with the framework’s flexibility and accuracy, will
benefit and ease future research on instance segmentation.
Finally, we showcase the generality of our framework
via the task of human pose estimation on the COCO keypoint dataset [23]. By viewing each keypoint as a one-hot
binary mask, with minimal modification Mask R-CNN can
be applied to detect instance-specific poses. Without tricks,
Mask R-CNN surpasses the winner of the 2016 COCO keypoint competition, and at the same time runs at 5 fps. Mask
R-CNN, therefore, can be seen more broadly as a flexible
framework for instance-level recognition and can be readily
extended to more complex tasks.


ビジョン・コミュニティは、短期間のうちに物体検出とセマンティック・セグメンテーションの結果を急速に改善してきました。このような進歩の大部分は
これらの進歩の大部分は，物体検出と意味分割のためのそれぞれのフレームワークであるFast/Faster RCNN [9, 29]やFully Convolutional Network (FCN) [24]などの強力なベースラインシステムによってもたらされました．
などの強力なベースラインシステムによって推進されています.
これらの手法は，概念的に直感的で，柔軟性とロバスト性に優れ，学習と推論の時間が短いという特徴があります．本研究の目的は
インスタンス・セグメンテーションのフレームワークを開発することです。
インスタンスのセグメンテーションが難しいのは
画像内のすべてのオブジェクトを正確に検出すると同時に
各インスタンスを正確にセグメント化する必要があるからです。そのため
セマンティックセグメンテーションでは、オブジェクトのインスタンスを区別することなく、各ピクセルを一定のカテゴリーに分類することが目的です。このように考えると、良い結果を得るためには、複雑な手法が必要になると考えられます。
このことから、良い結果を得るためには複雑な手法が必要であると考えられる。しかし、私たちは驚くほどシンプルで柔軟かつ高速なシステムが、これまでの最先端のインスタンス・セグメンテーションの結果を上回ることを示しています。
マスクR-CNNと呼ばれる我々の手法は、Faster R-CNN
マスクR-CNNと呼ばれる我々の手法は、Faster R-CNN[29]を拡張し、各関心領域(RoI)におけるセグメンテーション・マスク
この手法は、Faster R-CNN[29]を拡張したもので、分類とバウンディングボックス回帰の既存のブランチと並行して、各関心領域（RoI）のセグメンテーションマスクを予測するブランチを追加しています（図1）。マスクブランチは、各RoIに適用される小さなFCN
各RoIに適用され、ピクセル単位でセグメンテーションマスクを予測します。マスクR-CNNは、Faster R-CNNに比べて実装と学習が簡単です。
Faster R-CNNフレームワークは、実装と学習が簡単で
Faster R-CNNフレームワークは、幅広い柔軟なアーキテクチャ設計を可能にします。さらに。
さらに、マスクブランチはわずかな計算オーバーヘッドしか追加しません。
高速なシステムと迅速な実験を可能にしています。
原理的には、Mask R-CNNは、Faster R-CNNを直感的に拡張したものです。
Faster R-CNNを直感的に拡張したものですが、良い結果を得るためには、マスクブランチを適切に構築することが重要です。
は良い結果を得るためには重要です。最も重要なのは、Faster R-CNN
は、ネットワークの入力と出力の間のピクセル間の調整のために設計されていません。このことは、次のような点で明らかです。
RoIPool [14, 9]は、インスタンスにアテンダンスするための事実上のコア・オペレーションです。
このことは、RoIPool[14.9]が特徴抽出のために粗い空間量子化を行っていることからもわかります。このミスアライメントを修正するために、私たちはRoIAlignと呼ばれるシンプルで量子化を必要としないレイヤーを提案します。
正確な空間的位置を忠実に保持する。
マスクの精度を相対的に 10% から 50% 向上させることができます。
より厳しいローカライズメトリクスの下でより大きな利益を得ることができました。第二に、我々は
マスクとクラスの予測を切り離すことが重要であることを発見しました。
我々は マスクとクラスの予測を切り離すことが重要です。分類枝に依存してカテゴリーを予測します。これに対して
FCNでは通常、ピクセル単位でマルチクラスの分類を行います。
これは、セグメンテーションと分類を組み合わせたものです。
インスタンスのセグメンテーションには適していません。
マスクR-CNNは、これまでのシングルモデルでの結果を凌駕しています。
COCO
インスタンス・セグメンテーション・タスク[23]において、Mask R-CNNは、2016年のコンペティションの優勝者によるヘビーエンジニアリングされたエントリーを含む、過去の最先端のシングルモデルの結果をすべて上回りました。副産物として
副産物として、本手法はCOCOオブジェクト検出タスクにおいても優れています。
検出タスクにも優れています。アブレーションの実験では、複数の基本インスタンスを評価することで、そのロバスト性を実証し、コアの影響を分析することができます。
ロバスト性を実証し、コア・ファクターの影響を分析することができます。
私たちのモデルは、GPUを使って1フレームあたり約200msで動作します。
COCOの学習には、8GPUを搭載したマシン1台で1日から2日かかります。
COCOのトレーニングは、8GPUのマシン1台で1～2日で完了します。このトレーニングとテストの速さは、フレームワークの柔軟性と精度に加えて訓練とテストの速度が速いことは
インスタンス・セグメンテーションの将来の研究に役立つと確信しています。
最後に、我々のフレームワークの汎用性を、人間のポーズ推定のタスクを通して紹介します。
最後に、COCO keypoint dataset [23]における人間のポーズ推定タスクを通して、我々のフレームワークの一般性を紹介します。各キーポイントをワンショットのバイナリマスクと見なすことで
各キーポイントをワンショットのバイナリマスクと見なすことで、マスクR-CNNを最小限の修正で
インスタンス固有のポーズを検出することができます。トリックなしで
Mask R-CNNは、2016年のCOCO keypoint competitionの優勝者を凌駕し、同時に5fpsで動作しています。マスク
R-CNNは、インスタンスレベルの認識のための柔軟なフレームワークとして、より広い範囲で見ることができます。
インスタンスレベルの認識のための柔軟なフレームワークとして捉えることができます。
より複雑なタスクに拡張することができます。

# §2
R-CNN: The Region-based CNN (R-CNN) approach [10]
to bounding-box object detection is to attend to a manageable number of candidate object regions [33, 16] and evaluate convolutional networks independently on each RoI. R-CNN was extended [14, 9] to allow attending to RoIs
on feature maps using RoIPool, leading to fast speed and
better accuracy. Faster R-CNN [29] advanced this stream
by learning the attention mechanism with a Region Proposal Network (RPN). Faster R-CNN is flexible and robust
to many follow-up improvements (e.g., [30, 22, 17]), and is
the current leading framework in several benchmarks.
Driven by the effectiveness of RCNN, many approaches to instance segmentation are based
on segment proposals. Earlier methods [10, 12, 13, 6] resorted to bottom-up segments [33, 2]. DeepMask [27] and
following works [28, 5] learn to propose segment candidates, which are then classified by Fast R-CNN. In these
methods, segmentation precedes recognition, which is slow
and less accurate. Likewise, Dai et al. [7] proposed a complex multiple-stage cascade that predicts segment proposals from bounding-box proposals, followed by classification. Instead, our method is based on parallel prediction of
masks and class labels, which is simpler and more flexible.
Most recently, Li et al. [21] combined the segment proposal system in [5] and object detection system in [8] for
“fully convolutional instance segmentation” (FCIS). The
common idea in [5, 8, 21] is to predict a set of positionsensitive output channels fully convolutionally. These
channels simultaneously address object classes, boxes, and
masks, making the system fast. But FCIS exhibits systematic errors on overlapping instances and creates spurious
edges (Figure 5), showing that it is challenged by the fundamental difficulties of segmenting instances.

R-CNN リージョンベースCNN（R-CNN）のアプローチ [10]は
のアプローチは，管理可能な数のオブジェクト領域の候補に注目し[33, 16]，各RoIで独立して畳み込みネットワークを評価するというものである．R-CNN は，RoIPool を用いて，特徴マップ上の RoI
RoIPoolを用いて特徴量マップ上のRoIにアテンダンスできるように拡張され、高速かつ高精度な
精度が向上した。Faster R-CNN [29] は、この流れをさらに進めて
RPN(Region Proposal Network)を用いて注目メカニズムを学習することで、この流れを進化させた。Faster R-CNNは、柔軟でロバストです。
多くの改良（例：[30, 22, 17]）にも柔軟に対応しており
現在、いくつかのベンチマークでトップのフレームワークとなっています。
 RCNNの有効性を受けて、インスタンスセグメンテーションの多くのアプローチは、セグメント提案に基づいています。
初期の手法[10, 12, 13, 6]では、ボトムアップ型のセグメントに頼っていました[33, 2]。DeepMask [27] および
後続の作品 [28, 5] は、セグメントの候補を提案することを学習し、それを Fast R-CNN で分類している。これらの方法では
認識の前にセグメンテーションが行われるため、時間がかかり
精度が低くなります。同様に、Daiら[7]は、バウンディングボックスの提案からセグメントの提案を予測し、その後に分類するという複雑な多段カスケードを提案している。その代わり、我々の方法は、マスクとクラスラベルの並列予測に基づいています。
マスクとクラスラベルを並行して予測することで、よりシンプルで柔軟性の高い手法を実現しています。
最近では、Liら[21]が、[5]のセグメント提案方式と[8]の物体検出方式を組み合わせて
"fully convolutional instance segmentation" (FCIS)を開発した．ここでに共通するアイデアは，位置に依存する出力チャネルのセットを完全に畳み込みで予測することです
これらのチャンネルは、オブジェクトのクラス、ボックス、マスクを同時に処理します。
マスクを同時に扱うことで，システムを高速化しています．しかし，FCISは，オーバーラップするインスタンスで系統的なエラーが発生し，スプリアスなエッジを作成してしまいます（図5）．これは、インスタンスをセグメント化するという基本的な困難に直面していることを示しています。

# §3
Mask R-CNN is conceptually simple: Faster R-CNN has
two outputs for each candidate object, a class label and a
bounding-box offset; to this we add a third branch that outputs the object mask. Mask R-CNN is thus a natural and intuitive idea. But the additional mask output is distinct from
the class and box outputs, requiring extraction of much finer
spatial layout of an object. Next, we introduce the key elements of Mask R-CNN, including pixel-to-pixel alignment,
which is the main missing piece of Fast/Faster R-CNN.
Faster R-CNN: We begin by briefly reviewing the Faster
R-CNN detector [29]. Faster R-CNN consists of two stages.
The first stage, called a Region Proposal Network (RPN),
proposes candidate object bounding boxes. The second
stage, which is in essence Fast R-CNN [9], extracts features
using RoIPool from each candidate box and performs classification and bounding-box regression. The features used
by both stages can be shared for faster inference. We refer readers to [17] for latest, comprehensive comparisons
between Faster R-CNN and other frameworks.
Mask R-CNN: Mask R-CNN adopts the same two-stage
procedure, with an identical first stage (which is RPN). In
the second stage, in parallel to predicting the class and box
offset, Mask R-CNN also outputs a binary mask for each
RoI. This is in contrast to most recent systems, where classification depends on mask predictions (e.g. [27, 7, 21]).
Our approach follows the spirit of Fast R-CNN [9] that
applies bounding-box classification and regression in parallel (which turned out to largely simplify the multi-stage
pipeline of original R-CNN [10]).
 Formally, during training, we define a multi-task loss on
each sampled RoI as L = Lcls + Lbox + Lmask. The classification loss Lcls and bounding-box loss Lbox are identical as those defined in [9]. The mask branch has a Km2
-
dimensional output for each RoI, which encodes K binary
masks of resolution m × m, one for each of the K classes.
To this we apply a per-pixel sigmoid, and define Lmask as
the average binary cross-entropy loss. For an RoI associated with ground-truth class k, Lmask is only defined on the k-th
mask (other mask outputs do not contribute to the loss).
Our definition of Lmask allows the network to generate
masks for every class without competition among classes;
we rely on the dedicated classification branch to predict the class label used to select the output mask. This decouples
mask and class prediction. This is different from common
practice when applying FCNs [24] to semantic segmentation, which typically uses a per-pixel softmax and a multinomial cross-entropy loss. In that case, masks across classes
compete; in our case, with a per-pixel sigmoid and a binary
loss, they do not. We show by experiments that this formulation is key for good instance segmentation results.

Mask R-CNNは概念的にシンプルです。Faster R-CNNは
各候補オブジェクトに対して、クラスラベルとバウンディングボックスオフセットの2つの出力があります。
これに、オブジェクトマスクを出力する3つ目のブランチを加えます。マスクR-CNNは、このように自然で直感的なアイデアです。しかし、追加のマスク出力は、クラスとボックスの出力とは異なります。
クラスとボックスの出力とは別のもので、オブジェクトのより詳細な空間レイアウトを抽出する必要があります。
マスクR-CNNでは、ピクセル間のアライメントが重要な要素となります。
Faster R-CNN。まず，Faster R-CNN検出器[29]について簡単に説明します．
R-CNN検出器[29]について簡単に説明します．Faster R-CNNは2つのステージで構成されています。
最初のステージは、Region Proposal Network (RPN)と呼ばれます。
候補となるオブジェクトのバウンディングボックスを提案します。第二段階
第2段階は、Fast R-CNN [9]と呼ばれ、各候補ボックスからRoIPoolを用いて特徴を抽出します。
各候補ボックスからRoIPoolを用いて特徴量を抽出し，分類とバウンディングボックス回帰を行う．両ステージで使用される特徴量は
推論の高速化のために，両方のステージで使用される特徴を共有することができる．Faster R-CNNと他の手法との比較については，[17]を参照されたい．
Faster R-CNNと他のフレームワークとの最新の包括的な比較については、[17]を参照してください。
マスクR-CNN。マスクR-CNNは、同じ2段階の手順を採用しています。
第一段階（RPN）が同じである。では
第2段階では、クラスとボックスオフセットの予測と並行して
のオフセットを予測するのと並行して、マスクR-CNNは、各RoIのバイナリマスクを出力します。
RoIを出力します。これは、分類がマスク予測に依存している最近のほとんどのシステムとは対照的です（例：[27, 7, 21]）。
我々のアプローチは、Fast R-CNN [9]の精神に基づいています。
我々のアプローチは、バウンディングボックス分類と回帰を並行して行うFast R-CNN [9]の精神に従っています。
のパイプラインを大幅に簡素化できることがわかった）。
形式的には、トレーニング中に、サンプルされた各RoIのマルチタスク損失を、
L = Lcls + Lbox + Lmaskと定義する．分類損失Lclsとバウンディングボックス損失Lboxは，[9]で定義されたものと同じである．マスクブランチには，Km2
-
次元の出力を持ち，これは，解像度 m × m の K 個のバイナリ
マスクブランチは，各RoIに対してKm2次元の出力を持ち，K個のクラスに1つずつ，解像度m×mのK個のバイナリマスクをエンコードする．
これにピクセル毎のシグモイドを適用し，Lmaskを以下のように定義する．
平均的なバイナリクロスエントロピー損失と定義します。地面の真実のクラスkに関連するRoIに対して の場合、Lmaskはk番目のマスクに対してのみ定義されます。
マスクでのみ定義されます（他のマスク出力は損失に寄与しません）。
我々のLmaskの定義では、ネットワークがクラス間の競争なしに、すべてのクラスに対して
のマスクを生成することができます。
出力マスクの選択に使用されるクラスラベルの予測は、専用の分類ブランチに依存しています.これにより、マスク予測とクラス予測を切り離して
これは、一般的なFCN [24] をセマンティックセグメンテーションに適用する際の一般的な手法とは異なります。一般的には、ピクセル単位のソフトマックスと多項クロスエントロピー損失を使用します。この場合，クラスを超えたマスクが競合します．
競合しますが，我々の場合は，ピクセル単位のシグモイドとバイナリ損失の場合には、競合しません。我々の場合は、ピクセル単位のシグモイドと二値損失を用いることで、競合しない。

# §4
A mask encodes an input object’s
spatial layout. Thus, unlike class labels or box offsets
that are inevitably collapsed into short output vectors by
fully-connected (fc) layers, extracting the spatial structure
of masks can be addressed naturally by the pixel-to-pixel
correspondence provided by convolutions.
Specifically, we predict an m × m mask from each RoI using an FCN [24]. This allows each layer in the mask branch to maintain the explicit m × m object spatial layout without collapsing it into a vector representation that lacks spatial dimensions. Unlike previous methods that resort to fc layers for mask prediction [27, 28, 7], our fully convolutional representation requires fewer parameters, and is more accurate as demonstrated by experiments.
This pixel-to-pixel behavior requires our RoI features,
which themselves are small feature maps, to be well aligned
to faithfully preserve the explicit per-pixel spatial correspondence. This motivated us to develop the following
RoIAlign layer that plays a key role in mask prediction.

マスクは、入力されたオブジェクトの
空間的なレイアウトです。そのため、クラスラベルやボックスオフセットとは異なり
クラスラベルやボックスオフセットのように
マスクの空間構造を抽出することは、完全に接続された(fc)レイヤーによって
マスクの空間構造の抽出には、畳み込みによるピクセル間の
畳み込みによって得られるピクセル間の対応関係によって、マスクの空間構造を自然に抽出することができます。
具体的には，FCN[24]を用いて，各RoIからm×mのマスクを予測する．これにより，マスクブランチの各層は，明示的な m × m のオブジェクトの空間レイアウトを，空間的な次元を持たないベクトル表現に折り畳むことなく維持することができる．マスク予測のためにfcレイヤーに頼る従来の手法[27, 28, 7]とは異なり，我々の完全な畳み込み表現は，より少ないパラメータしか必要とせず，実験で実証されたように，より正確である．
このピクセル間の動作は、RoI機能を必要とします。
それ自体が小さなフィーチャーマップであるRoIフィーチャは、ピクセルごとの空間的な対応関係を忠実に維持するために
画素ごとの明示的な空間的対応関係を忠実に維持するためには そのために、次のような開発を行いました。
マスク予測に重要な役割を果たすRoIAlignレイヤー。