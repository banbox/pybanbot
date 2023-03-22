# 分析方法
预期价格阈值是0.4%

# 指标搭配研究
## 布林带
计算布林带相关因子：布林带方向，归一化宽度，布林带强度。
目前使用ema_3作为布林带计算对象，起到一定的平滑作用
bb20_facwid+bb20_sub1+bb20_sub3的取值、胜率、总次数
  pos:   26451 / 38865
  neg:   57126 / 90523
## 移动均线MA
移动平均指标，包含：EMA,SMA
对于未来第3个蜡烛利润，使用close-sma3效果最优。
测试了：close-ema3,ema3-ema5,ema3-sma5等。
sma3_chg:25  thres:0.3
  pos:   225477 / 338863
  neg:   338956 / 500560
sma3_chg:9  thres:0.4
  pos:   62252 / 88710
  neg:   181467 / 266688
sma3_chg:11  thres:0.4
  pos:   42908 / 59757
  neg:   116010 / 160777
sma3_chg:13  thres:0.4
  pos:   77678 / 112551
  neg:   227042 / 346727
sma3_chg:15  thres:0.4
  pos:   204981 / 334965
  neg:   185174 / 273087
sma3_chg:17  thres:0.4
  pos:   95642 / 141458
  neg:   195567 / 291106
sma3_chg:19  thres:0.4
  pos:   145729 / 226057
  neg:   143997 / 205083
sma3_chg:25  thres:0.4
  pos:   95899 / 141780
  neg:   137069 / 194156
## CMF
CMF指标分析，在横盘震荡期间效果不好，所以要结合NATR
atr:11&atr_sub7:5 + sma3_chg:15 + raw_cmf:11&cmf_sub3:5  thres:0.4
  pos:   146588 / 224463
  neg:   185141 / 270415
atr:11|atr_sub7:5 + sma3_chg:15 + raw_cmf:11|cmf_sub3:5|r7max_d:5  thres:0.4
  pos:   141501 / 214252
  neg:   181861 / 265131
close_s_sma3|r7max_d|cmf_sub1
acc:   49210 / 66218
acc:   537116 / 865663

close_s_sma3|cmf_sub1
acc:   48775 / 65484
acc:   536964 / 865663

close_s_sma3
acc:   49581 / 66912
acc:   536711 / 865663


atr|atr_sub7|cmf_sub1
acc:   7184 / 11709
acc:   23279 / 36912

atr|atr_sub7|cmf_sub3
acc:   7679 / 12666
acc:   31489 / 48455

atr|atr_sub7|cmf_sube3
acc:   1228 / 2041
acc:   25134 / 39698

atr|atr_sub7|cmf|cmf_sub3
acc:   13628 / 21362
acc:   41962 / 64956


close_s_sma3|cmf|cmf_sub3
acc:   46751 / 65023
acc:   66719 / 106447

close_s_sma3|r7max_d|cmf|cmf_sub3
acc:   46457 / 64416
acc:   72929 / 116923

atr|atr_sub7|close_s_sma3|cmf|cmf_sub3
  pos:   51054 / 71207
  neg:   151648 / 235292

atr|atr_sub7|close_s_sma3|cmf|cmf_sub1
  pos:   50730 / 70908
  neg:   147183 / 229793

atr|atr_sub7|close_s_sma3|cmf
  pos:   46862 / 65140
  neg:   141312 / 225944

atr|atr_sub7|close_s_sma3|r7max_d|cmf|cmf_sub3
  pos:   55056 / 77657
  neg:   157494 / 242691



atr|atr_sub7|cmf|over_100max|cmf_sub1
  pos:   22026 / 33062
  neg:   31609 / 47821
atr|atr_sub7|cmf|below_100min|cmf_sub1
  pos:   19203 / 29028
  neg:   38597 / 58496
atr|atr_sub7|cmf|cmf_sub1
  pos:   14305 / 21967
  neg:   24638 / 38224
atr|atr_sub7|cmf|cmf_sub3
  pos:   7070 / 10854
  neg:   25688 / 39824
## CRSI
atr:11|atr_sub7:5 + raw_crsi:33  thres:0.4
  pos:   362 / 512
  neg:   49309 / 79220
## EWO  Elliot Wave Oscillator
raw_ewo:9|ewo_sub1:5|ewo_sub3:5  thres:0.4
  pos:   113435 / 175262
  neg:   227024 / 370686
raw_ewo:11|ewo_sub1:5|ewo_sub3:5  thres:0.4【使用】
  pos:   114901 / 177963
  neg:   228812 / 373675
raw_ewo:13|ewo_sub1:5|ewo_sub3:5  thres:0.4
  pos:   110938 / 170884
  neg:   215626 / 350503
raw_ewo:15|ewo_sub1:5|ewo_sub3:5  thres:0.4
  pos:   113118 / 174611
  neg:   218075 / 354933
raw_ewo:11|ewo_sub1:5|ewo_sub3:7  thres:0.4
  pos:   108130 / 165845
  neg:   216116 / 351229
## KAMA
KAMA趋势均值指标，跟SMA，EWO等类似，需要评估哪个更好
km_fac:9|km_sub1:5|km_sub10:5  thres:0.4
  pos:   36606 / 53954
  neg:   128777 / 207289
km_fac:9|km_sub1:5|km_sub10:5  thres:0.3
  pos:   38251 / 54222
  neg:   204440 / 308739
km_fac:9|km_sub1:5|km_sub10_s2:5  thres:0.4
  pos:   37005 / 54529
  neg:   177622 / 291751
## Stoch RSI
基于RSI的震荡指标。对做多信号和做空信号，均有较好检出效果。
atr:11|atr_sub7:5 + srsi_k:21|srsi_k_sub_d:5  thres:0.4
  pos:   35845 / 56247
  neg:   69441 / 107836
atr:11|atr_sub7:5 + srsi_k:21|srsi_k_sub_d:5  thres:0.3
  pos:   74793 / 117431
  neg:   136291 / 212224
atr:11|atr_sub7:5 + srsi_k:21|srsi_k_sub_d:5  thres:0.3
  pos:   73769 / 115519
  neg:   149705 / 235205
cluster srsi_k_sub_d cost  9.84s, [-14, -5, 0, 6, 14]
concat cols: ['srsi_k', 'srsi_k_sub_d'] cost 26.4s
cluster atr cost  0.71s, [5, 9, 12, 16, 20, 25, 31, 39, 49, 64, 88]
cluster atr_sub7 cost  0.43s, [-5, -1, 2, 10, 34]
concat cols: ['atr', 'atr_sub7'] cost 25.2s
## MFI
MFI：资金流动指数。和CMF类似，是交易量加权的价格指数。
这个指标据说很难用，效果不佳
短期指标，预测后3个
sma3_chg:15 + raw_mfi:15|mfi_sub1:5  thres:0.4
  pos:   140787 / 219803
  neg:   184715 / 272512
sma3_chg:15 + atr:11|atr_sub7:5 + raw_mfi:15|mfi_sub1:5  thres:0.4
  pos:   145179 / 221762
  neg:   180034 / 262250
raw_ewo:11|ewo_sub1:5|ewo_sub3:5 + raw_mfi:15|mfi_sub1:5  thres:0.4
  pos:   104901 / 158562
  neg:   186857 / 295228
km_fac:9|km_sub1:5|km_sub10:5 + raw_mfi:15|mfi_sub1:5  thres:0.4
  pos:   37067 / 54106
  neg:   135555 / 211829
## WR  williams %R
sma3_chg:15 + atr:11|atr_sub7:5 + raw_wr:11  thres:0.4
  pos:   155160 / 242189
  neg:   184754 / 271772
sma3_chg:15 + raw_wr:11  thres:0.4
  pos:   145579 / 228733
  neg:   184983 / 272868
sma10_chg:9 + atr:11|atr_sub7:5 + raw_wr:11  thres:0.4
  pos:   26158 / 40991
  neg:   34059 / 52740
bb20_sub1:5|bb20_sub3:5|bb20_facwid:11 + raw_wr:11  thres:0.4
  pos:   31548 / 46064
  neg:   71136 / 111522
rsi_25:15|rsi_25_sub1:5 + raw_wr:11  thres:0.4
  pos:   10972 / 16971
  neg:   10027 / 14271
## RMI 
震荡指标，结合SMA对短期市场的做多和做空均有良好检测效果
sma3_chg:15 + raw_rmi:15|rmi_sub1:5  thres:0.4
  pos:   147979 / 230381
  neg:   170289 / 246986
sma3_chg:15 + atr:11|atr_sub7:5 + raw_rmi:15|rmi_sub1:5  thres:0.4【使用】
  pos:   148322 / 225838
  neg:   181037 / 261601
sma10_chg:9 + raw_rmi:15|rmi_sub1:5  thres:0.4
  pos:   47495 / 72568
  neg:   78093 / 122460
sma10_chg:9 + atr:11|atr_sub7:5 + raw_rmi:15|rmi_sub1:5  thres:0.4
  pos:   67446 / 102416
  neg:   99686 / 151001
## RSI 
和RMI结果有大部分重合
sma3_chg:15 + rsi_25:15|rsi_25_sub1:5  thres:0.4
  pos:   144552 / 225216
  neg:   185496 / 273766
sma3_chg:15 + atr:11|atr_sub7:5 + rsi_25:15|rsi_25_sub1:5  thres:0.4【使用】
  pos:   150774 / 232033
  neg:   182446 / 265251
## KDJ
atr:11|atr_sub7:5 + kdj_k_sub1:5|kdj_j_d:5|slowk:13  thres:0.4
  pos:   29256 / 45829
  neg:   54369 / 83004
atr|atr_sub7|kdj_k_sub1|kdj_j_d|slowk
  pos:   29256 / 45829
  neg:   54369 / 83004

atr=19,atr_sub=7
acc:   14640 / 21856
acc:   33245 / 49998


atr=11,atr_sub=7
acc:   14780 / 22337
acc:   31746 / 47815


atr=11,atr_sub=5
acc:   15230 / 23128
acc:   28791 / 42853

atr=9,atr_sub=5
acc:   11149 / 16611
acc:   29441 / 44363

atr=9,atr_sub=5(upper=100),kdj_j_d=7
acc:   14958 / 22736
acc:   31529 / 47749


atr=9,atr_sub=5(upper=100),kdj_j_d=5,slowk=15
acc:   15208 / 23650
acc:   28882 / 43382

atr=9,atr_sub=5(upper=100),kdj_j_d=5,slowk=11
acc:   12901 / 19831
acc:   28261 / 42445

atr=9,atr_sub=5(upper=100),kdj_j_d=5,slowk=13
acc:   14647 / 22686
acc:   28418 / 42691



atr_sub5
acc:   12177 / 18610
acc:   29119 / 43503

atr_sub7
acc:   12409 / 18959
acc:   28342 / 42590

atr_sub9
acc:   12075 / 18260
acc:   28811 / 43468

atr_sub10
acc:   10996 / 16800
acc:   28105 / 42522

atr_sub15
acc:   7709 / 11834
acc:   25127 / 38263

atr_sqrt|atr_sub10
acc:   12482 / 18835
acc:   31384 / 47889


atr_sqrt|atr_sub7|atr_9
acc:   14842 / 22866
acc:   30785 / 46030


atr_sqrt|atr_sub7|atr_11
acc:   17085 / 26350
acc:   30511 / 45353
## AROON
rsi_25:15|rsi_25_sub1:5 + arron_up:11|aroon_down:11  thres:0.4
  pos:   13334 / 20884
  neg:   15337 / 22701
