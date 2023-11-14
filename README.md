# Goppa Code Decorder by Peterson Algorithm over Prime Fields
gcc -O3 fujitama.c

# 20231114
小休止。
なんでこれを作ったか。
よく、あるものを使えばいいと言われるのだが、そういう堕落した人のほうが多いのだろう。
作っても不完全で、安全なライブラリもアプリもできないから、プロの作ったものを使えばいいということだ。
でも自分としては、ただ上意下達のごとく使うだけの人になるのではなく、それがどのような仕組みでできているのかとか、なぜそれは安全だと言えるのかを理解したい。

それ抜きにこういうプログラムは書けないと思うし、そのほうが寧ろ健全な気がするのだ。
そしてそれを理解するためには、自分の手を動かして作るのが一番いいと思っている。
紙の上だけで考えることもできるけど、それだけでは実感としてイメージを把握できないので、理論がどのように実装に反映されるのかというのも関係があると思う。
これを公開しているのは、公開しないとログインできないときにクローンすることができないからだ。

復号法のプログラムとしての終点は、誤りの値を計算するところまでにするつもりだ。


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
