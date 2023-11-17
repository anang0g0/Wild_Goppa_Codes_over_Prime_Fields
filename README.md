# Goppa Code Decorder by Peterson Algorithm over Prime Fields
gcc -O3 fujiyama.c  
ulimit -s unlimited  
./a.out  

-------
# 20231117
vpowmodが原因で時間がかかっている。
冪剰余演算さえ早くできれば、既にsagemathに匹敵する速度にはなっていると思う。

もしこれが実現すれば、最小多項式に依存しないランダム既約多項式を素早く決定できるというのに。

素数を重みの軽い素数に変更。
ウィンドウ法も考えたけど、多項式が変わるたびにテーブルを作り直さないといけないので逆に遅くなりそう。
何間違っているのかも知れないが。

次の目標を電子署名に選定。
PKP使った電子署名は昔からある割に、ブラインドバージョンがないのでやってみるつもり。

これにて終了。

# 20231116
シルベスター行列とか使って共通因子の存在だけを判定する方法を追加した。
行列式を計算するだけなのですが遅いです。
GCDを計算しない分早くなるかと期待していたのですが、GCDを計算しなくてもいい方法は初めて知りました。

ChatGPTに早くなるか聞いてみます。

まあ、欲張ってGPTに聞いてみたのですが、一発で動かないのでしびれを切らしました。
何が悪いのだろう、自分の関数だと正の数になるのに、GPTバージョンはなぜか０。


# 20231115
誤り値の決定。これはちょっと難しいぞと思って３日くらい悩んだけど、よく考えてみれば生成行列にエラーの値をかけただけだから、
エラーの位置が決まったら、そこの列を取り出して部分行列の逆行列をシンドロームにかけてやればいいとわかってスッキリ。
寧ろエラーの位置を特定するほうが凄いかも。（なので誤り値決定バージョンは公開しません。）

このアルゴリズムは、符号の生成多項式なしにシンドロームだけで復号できるので、BM法に近いのかも知れない。

だいたいやりたいことは終わったので、次の目標はPermte Kernel Problemを使った電子署名に挑戦したいです。

つまり、次の目標は電子署名という感じです。
アメリカも新型電子署名を補修してるみたいだし、選ばれないと思うけどやってみよう。
電子署名は、暗号と違って北朝鮮に悪用されることもできないだろう。
そもそも私はブロックチェーンの技術に興味があるので、量子体制のある電子署名法は一つの理想を実現することと同義なのだ。

# 20231114
小休止。
なんでこれを作ったか。
よく、あるものを使えばいいと言われるのだが、そういう堕落した人のほうが多いのだろう。
作っても不完全で、安全なライブラリもアプリもできないから、プロの作ったものを使えばいいということだ。
でも自分としては、ただ上意下達のごとく使うだけの人になるのではなく、それがどのような仕組みでできているのかとか、なぜそれは安全だと言えるのかを理解したい。

それ抜きにこういうプログラムは書けないと思うし、そのほうが寧ろ健全な気がするのだ。
そしてそれを理解するためには、自分の手を動かして作るのが一番いいと思っている。
紙の上だけで考えることもできるけど、それだけでは実感としてイメージを把握できないので、理論がどのように実装に反映されるのかというのも確かめたい。
これを公開しているのは、公開しないとログインできないときにクローンすることができないからだ。

~~復号法のプログラムとしての終点は、誤りの値を計算するところまでにするつもりだ。~~


# 20231113
もっと早くなるんじゃないかと思って、gcdの代わりにシルベスター行列とか使って最大公約多項式を計算しようとしたのだが、
結局問題はgcdではなくpowmodだったので多項式の次数の上限を下げたら効果的に早くなった。

なぜsagemathはあんなに瞬間で判別できるのかが謎。
一体何のアルゴリズムを採用しているのかわからないが、同じくらいの速度のプログラムにしたい。

終結式使ったGCDとユークリッドを使ったGCDでは何がどう違うのかとか、どちらがメリットがあるのかとか色々興味があります。
あと、素数というと数論的に色々キレイらしいので、素体上でGoppa符号を考えると、どのようなことが言えるのか、何かいいことが見るかるかも知れないとちょっと気にしてみたり。

やる気と集中力があったらやります。

# 20231112
現在符号の鍵生成をマルチプロセスを使った並列化とパターソン復号に取り組んでいます。

そしてピーターソン復号法も並列化の予定です。
ループを部分に分解して計算します。
一番単純な並列化です。

最近はスマホのＡＲＭでさえマルチコアですので、比較的新しめのハードで並列化できるようにするつもりです。
もちろんシングルタスクでも動きます。

だがもう寝るｗ

# 20231111
とりあえず面倒なこと一切抜きにして、一番単純な答えがこれだろうと思って公開しました。  
素体上で動かすとどういう訳か早いです。

拡大体を扱うための議論が必要なくなるというのは結構楽です。
今後この符号の性能や、高速な復号手順を調べていきたいと思います。

実装にあたり、ピーターソン復号法は以下の参考文献を見ました。
ほかの復号法も攻略できるかも。


参考文献：代数系と符号理論（オーム社）
