û5
Í£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18Ó-

contrastive_cnn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*/
shared_name contrastive_cnn/dense_2/kernel

2contrastive_cnn/dense_2/kernel/Read/ReadVariableOpReadVariableOpcontrastive_cnn/dense_2/kernel*
_output_shapes
:	@*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
: *
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
: *
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
: *
dtype0

conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
: *
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
: *
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
: *
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
: *
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
: *
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
: *
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:@*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:@*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:@*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:@*
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes
:@*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes
:@*
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_13/kernel
~
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*'
_output_shapes
:@*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:*
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
:*
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes	
:*
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
¢
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
: *
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
: *
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
: *
dtype0
¤
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
: *
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:@*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes
:@*
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
:*
dtype0

NoOpNoOp
ì
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¦
valueB B

feature_extractor
head
	variables
regularization_losses
trainable_variables
	keras_api

signatures
k

cnn_blocks
	gba

	variables
regularization_losses
trainable_variables
	keras_api
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
Î
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
/28
029
130
231
332
433
534
635
736
837
938
:39
;40
<41
42
 
Þ
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
28
­
	variables
=layer_metrics

>layers
?non_trainable_variables
regularization_losses
@metrics
Alayer_regularization_losses
trainable_variables
 
1
B0
C1
D2
E3
F4
G5
H6
R
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
Æ
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
/28
029
130
231
332
433
534
635
736
837
938
:39
;40
<41
 
Ö
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
­

	variables
Mlayer_metrics

Nlayers
Onon_trainable_variables
regularization_losses
Pmetrics
Qlayer_regularization_losses
trainable_variables
ZX
VARIABLE_VALUEcontrastive_cnn/dense_2/kernel&head/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
	variables
Rlayer_metrics

Slayers
Tnon_trainable_variables
regularization_losses
Umetrics
Vlayer_regularization_losses
trainable_variables
KI
VARIABLE_VALUEconv2d_7/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_7/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_7/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_7/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_8/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_8/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_8/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_8/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_9/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_9/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_9/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_9/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_10/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_10/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_10/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_10/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_11/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_11/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_11/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_11/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_12/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_12/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_12/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_12/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2d_13/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_13/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_13/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_13/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_7/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_7/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_8/moving_mean'variables/30/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_8/moving_variance'variables/31/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_9/moving_mean'variables/32/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_9/moving_variance'variables/33/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_10/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_10/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_11/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_11/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_12/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_12/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_13/moving_mean'variables/40/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_13/moving_variance'variables/41/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
f
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
 
 
­
Wlayer_with_weights-0
Wlayer-0
Xlayer_with_weights-1
Xlayer-1
Ylayer-2
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
­
^layer_with_weights-0
^layer-0
_layer_with_weights-1
_layer-1
`layer-2
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
­
elayer_with_weights-0
elayer-0
flayer_with_weights-1
flayer-1
glayer-2
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
­
llayer_with_weights-0
llayer-0
mlayer_with_weights-1
mlayer-1
nlayer-2
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
­
slayer_with_weights-0
slayer-0
tlayer_with_weights-1
tlayer-1
ulayer-2
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
®
zlayer_with_weights-0
zlayer-0
{layer_with_weights-1
{layer-1
|layer-2
}	variables
~regularization_losses
trainable_variables
	keras_api
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
²
I	variables
layer_metrics
layers
non_trainable_variables
Jregularization_losses
metrics
 layer_regularization_losses
Ktrainable_variables
 
8
B0
C1
D2
E3
F4
G5
H6
	7
f
/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13
 
 
 
 
 
 
 

_inbound_nodes

kernel
bias
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
Ç
_inbound_nodes
	axis
	gamma
beta
/moving_mean
0moving_variance
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
k
_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
*
0
1
2
3
/4
05
 

0
1
2
3
²
Z	variables
layer_metrics
 layers
¡non_trainable_variables
[regularization_losses
¢metrics
 £layer_regularization_losses
\trainable_variables

¤_inbound_nodes

kernel
bias
¥_outbound_nodes
¦	variables
§regularization_losses
¨trainable_variables
©	keras_api
Ç
ª_inbound_nodes
	«axis
	gamma
beta
1moving_mean
2moving_variance
¬_outbound_nodes
­	variables
®regularization_losses
¯trainable_variables
°	keras_api
k
±_inbound_nodes
²	variables
³regularization_losses
´trainable_variables
µ	keras_api
*
0
1
2
3
14
25
 

0
1
2
3
²
a	variables
¶layer_metrics
·layers
¸non_trainable_variables
bregularization_losses
¹metrics
 ºlayer_regularization_losses
ctrainable_variables

»_inbound_nodes

kernel
bias
¼_outbound_nodes
½	variables
¾regularization_losses
¿trainable_variables
À	keras_api
Ç
Á_inbound_nodes
	Âaxis
	gamma
beta
3moving_mean
4moving_variance
Ã_outbound_nodes
Ä	variables
Åregularization_losses
Ætrainable_variables
Ç	keras_api
k
È_inbound_nodes
É	variables
Êregularization_losses
Ëtrainable_variables
Ì	keras_api
*
0
1
2
3
34
45
 

0
1
2
3
²
h	variables
Ílayer_metrics
Îlayers
Ïnon_trainable_variables
iregularization_losses
Ðmetrics
 Ñlayer_regularization_losses
jtrainable_variables

Ò_inbound_nodes

kernel
 bias
Ó_outbound_nodes
Ô	variables
Õregularization_losses
Ötrainable_variables
×	keras_api
Ç
Ø_inbound_nodes
	Ùaxis
	!gamma
"beta
5moving_mean
6moving_variance
Ú_outbound_nodes
Û	variables
Üregularization_losses
Ýtrainable_variables
Þ	keras_api
k
ß_inbound_nodes
à	variables
áregularization_losses
âtrainable_variables
ã	keras_api
*
0
 1
!2
"3
54
65
 

0
 1
!2
"3
²
o	variables
älayer_metrics
ålayers
ænon_trainable_variables
pregularization_losses
çmetrics
 èlayer_regularization_losses
qtrainable_variables

é_inbound_nodes

#kernel
$bias
ê_outbound_nodes
ë	variables
ìregularization_losses
ítrainable_variables
î	keras_api
Ç
ï_inbound_nodes
	ðaxis
	%gamma
&beta
7moving_mean
8moving_variance
ñ_outbound_nodes
ò	variables
óregularization_losses
ôtrainable_variables
õ	keras_api
k
ö_inbound_nodes
÷	variables
øregularization_losses
ùtrainable_variables
ú	keras_api
*
#0
$1
%2
&3
74
85
 

#0
$1
%2
&3
²
v	variables
ûlayer_metrics
ülayers
ýnon_trainable_variables
wregularization_losses
þmetrics
 ÿlayer_regularization_losses
xtrainable_variables

_inbound_nodes

'kernel
(bias
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
Ç
_inbound_nodes
	axis
	)gamma
*beta
9moving_mean
:moving_variance
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
k
_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
*
'0
(1
)2
*3
94
:5
 

'0
(1
)2
*3
²
}	variables
layer_metrics
layers
non_trainable_variables
~regularization_losses
metrics
 layer_regularization_losses
trainable_variables

_inbound_nodes

+kernel
,bias
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
Ç
_inbound_nodes
	axis
	-gamma
.beta
;moving_mean
<moving_variance
_outbound_nodes
 	variables
¡regularization_losses
¢trainable_variables
£	keras_api
k
¤_inbound_nodes
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
*
+0
,1
-2
.3
;4
<5
 

+0
,1
-2
.3
µ
	variables
©layer_metrics
ªlayers
«non_trainable_variables
regularization_losses
¬metrics
 ­layer_regularization_losses
trainable_variables
 
 
 
 
 
 
 

0
1
 

0
1
µ
	variables
®layer_metrics
¯layers
°non_trainable_variables
regularization_losses
±metrics
 ²layer_regularization_losses
trainable_variables
 
 
 

0
1
/2
03
 

0
1
µ
	variables
³layer_metrics
´layers
µnon_trainable_variables
regularization_losses
¶metrics
 ·layer_regularization_losses
trainable_variables
 
 
 
 
µ
	variables
¸layer_metrics
¹layers
ºnon_trainable_variables
regularization_losses
»metrics
 ¼layer_regularization_losses
trainable_variables
 

W0
X1
Y2

/0
01
 
 
 
 

0
1
 

0
1
µ
¦	variables
½layer_metrics
¾layers
¿non_trainable_variables
§regularization_losses
Àmetrics
 Álayer_regularization_losses
¨trainable_variables
 
 
 

0
1
12
23
 

0
1
µ
­	variables
Âlayer_metrics
Ãlayers
Änon_trainable_variables
®regularization_losses
Åmetrics
 Ælayer_regularization_losses
¯trainable_variables
 
 
 
 
µ
²	variables
Çlayer_metrics
Èlayers
Énon_trainable_variables
³regularization_losses
Êmetrics
 Ëlayer_regularization_losses
´trainable_variables
 

^0
_1
`2

10
21
 
 
 
 

0
1
 

0
1
µ
½	variables
Ìlayer_metrics
Ílayers
Înon_trainable_variables
¾regularization_losses
Ïmetrics
 Ðlayer_regularization_losses
¿trainable_variables
 
 
 

0
1
32
43
 

0
1
µ
Ä	variables
Ñlayer_metrics
Òlayers
Ónon_trainable_variables
Åregularization_losses
Ômetrics
 Õlayer_regularization_losses
Ætrainable_variables
 
 
 
 
µ
É	variables
Ölayer_metrics
×layers
Ønon_trainable_variables
Êregularization_losses
Ùmetrics
 Úlayer_regularization_losses
Ëtrainable_variables
 

e0
f1
g2

30
41
 
 
 
 

0
 1
 

0
 1
µ
Ô	variables
Ûlayer_metrics
Ülayers
Ýnon_trainable_variables
Õregularization_losses
Þmetrics
 ßlayer_regularization_losses
Ötrainable_variables
 
 
 

!0
"1
52
63
 

!0
"1
µ
Û	variables
àlayer_metrics
álayers
ânon_trainable_variables
Üregularization_losses
ãmetrics
 älayer_regularization_losses
Ýtrainable_variables
 
 
 
 
µ
à	variables
ålayer_metrics
ælayers
çnon_trainable_variables
áregularization_losses
èmetrics
 élayer_regularization_losses
âtrainable_variables
 

l0
m1
n2

50
61
 
 
 
 

#0
$1
 

#0
$1
µ
ë	variables
êlayer_metrics
ëlayers
ìnon_trainable_variables
ìregularization_losses
ímetrics
 îlayer_regularization_losses
ítrainable_variables
 
 
 

%0
&1
72
83
 

%0
&1
µ
ò	variables
ïlayer_metrics
ðlayers
ñnon_trainable_variables
óregularization_losses
òmetrics
 ólayer_regularization_losses
ôtrainable_variables
 
 
 
 
µ
÷	variables
ôlayer_metrics
õlayers
önon_trainable_variables
øregularization_losses
÷metrics
 ølayer_regularization_losses
ùtrainable_variables
 

s0
t1
u2

70
81
 
 
 
 

'0
(1
 

'0
(1
µ
	variables
ùlayer_metrics
úlayers
ûnon_trainable_variables
regularization_losses
ümetrics
 ýlayer_regularization_losses
trainable_variables
 
 
 

)0
*1
92
:3
 

)0
*1
µ
	variables
þlayer_metrics
ÿlayers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
 
 
 
 
µ
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
 

z0
{1
|2

90
:1
 
 
 
 

+0
,1
 

+0
,1
µ
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
 
 
 

-0
.1
;2
<3
 

-0
.1
µ
 	variables
layer_metrics
layers
non_trainable_variables
¡regularization_losses
metrics
 layer_regularization_losses
¢trainable_variables
 
 
 
 
µ
¥	variables
layer_metrics
layers
non_trainable_variables
¦regularization_losses
metrics
 layer_regularization_losses
§trainable_variables
 

0
1
2

;0
<1
 
 
 
 
 
 
 
 
 

/0
01
 
 
 
 
 
 
 
 
 
 
 
 
 
 

10
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 

30
41
 
 
 
 
 
 
 
 
 
 
 
 
 
 

50
61
 
 
 
 
 
 
 
 
 
 
 
 
 
 

70
81
 
 
 
 
 
 
 
 
 
 
 
 
 
 

90
:1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 
 
 
 

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_variancecontrastive_cnn/dense_2/kernel*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*8
config_proto(&

CPU

GPU2*0J

   E8 *.
f)R'
%__inference_signature_wrapper_6972390
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Û
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2contrastive_cnn/dense_2/kernel/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOpConst*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *)
f$R"
 __inference__traced_save_6976879
ú
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecontrastive_cnn/dense_2/kernelconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/betaconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/betaconv2d_9/kernelconv2d_9/biasbatch_normalization_9/gammabatch_normalization_9/betaconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/betaconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/betaconv2d_12/kernelconv2d_12/biasbatch_normalization_12/gammabatch_normalization_12/betaconv2d_13/kernelconv2d_13/biasbatch_normalization_13/gammabatch_normalization_13/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance"batch_normalization_10/moving_mean&batch_normalization_10/moving_variance"batch_normalization_11/moving_mean&batch_normalization_11/moving_variance"batch_normalization_12/moving_mean&batch_normalization_12/moving_variance"batch_normalization_13/moving_mean&batch_normalization_13/moving_variance*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *,
f'R%
#__inference__traced_restore_6977018ãû*
«
Ç
-__inference_cnn_block_5_layer_call_fn_6975456
conv2d_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69704002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameconv2d_12_input
 

H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974795
conv2d_9_input+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_9/AssignNewValue¢&batch_normalization_9/AssignNewValue_1°
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_9/Conv2D/ReadVariableOpÇ
conv2d_9/Conv2DConv2Dconv2d_9_input&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_9/Conv2D§
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp¬
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_9/BiasAdd¶
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp¼
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1é
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_9/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_9/FusedBatchNormV3
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1
re_lu_9/ReluRelu*batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_9/ReluÆ
IdentityIdentityre_lu_9/Relu:activations:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_9_input
Ò
`
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6975937

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Õ 

H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975483
conv2d_13_input,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource
identity¢%batch_normalization_13/AssignNewValue¢'batch_normalization_13/AssignNewValue_1´
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_13/Conv2D/ReadVariableOpÌ
conv2d_13/Conv2DConv2Dconv2d_13_input'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_13/Conv2D«
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp±
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_13/BiasAddº
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_13/ReadVariableOpÀ
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_13/ReadVariableOp_1í
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1û
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_13/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_13/FusedBatchNormV3
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_13/AssignNewValue
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_13/AssignNewValue_1
re_lu_13/ReluRelu+batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_13/ReluÊ
IdentityIdentityre_lu_13/Relu:activations:0&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_1:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameconv2d_13_input
¿
F
*__inference_re_lu_12_layer_call_fn_6976570

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_12_layer_call_and_return_conditional_losses_69703142
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ
«
8__inference_batch_normalization_11_layer_call_fn_6976403

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_69699602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦
Æ
-__inference_cnn_block_2_layer_call_fn_6974837
conv2d_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69694252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_9_input
®
«
8__inference_batch_normalization_12_layer_call_fn_6976496

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_69701952
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ 

H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975225
conv2d_11_input,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identity¢%batch_normalization_11/AssignNewValue¢'batch_normalization_11/AssignNewValue_1³
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_11/Conv2D/ReadVariableOpË
conv2d_11/Conv2DConv2Dconv2d_11_input'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_11/Conv2Dª
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp°
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_11/BiasAdd¹
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOp¿
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1ì
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ö
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_11/FusedBatchNormV3
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValue
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1
re_lu_11/ReluRelu+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_11/ReluÉ
IdentityIdentityre_lu_11/Relu:activations:0&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_1:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)
_user_specified_nameconv2d_11_input
Ë
°
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976138

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6968690

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
ª
7__inference_batch_normalization_8_layer_call_fn_6975932

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69690212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§
­
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6975795

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976452

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
²
«
8__inference_batch_normalization_13_layer_call_fn_6976717

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_69705082
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º'
æ
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6972208

inputs#
feature_extractor_cnn_1_6972119#
feature_extractor_cnn_1_6972121#
feature_extractor_cnn_1_6972123#
feature_extractor_cnn_1_6972125#
feature_extractor_cnn_1_6972127#
feature_extractor_cnn_1_6972129#
feature_extractor_cnn_1_6972131#
feature_extractor_cnn_1_6972133#
feature_extractor_cnn_1_6972135#
feature_extractor_cnn_1_6972137#
feature_extractor_cnn_1_6972139#
feature_extractor_cnn_1_6972141#
feature_extractor_cnn_1_6972143#
feature_extractor_cnn_1_6972145#
feature_extractor_cnn_1_6972147#
feature_extractor_cnn_1_6972149#
feature_extractor_cnn_1_6972151#
feature_extractor_cnn_1_6972153#
feature_extractor_cnn_1_6972155#
feature_extractor_cnn_1_6972157#
feature_extractor_cnn_1_6972159#
feature_extractor_cnn_1_6972161#
feature_extractor_cnn_1_6972163#
feature_extractor_cnn_1_6972165#
feature_extractor_cnn_1_6972167#
feature_extractor_cnn_1_6972169#
feature_extractor_cnn_1_6972171#
feature_extractor_cnn_1_6972173#
feature_extractor_cnn_1_6972175#
feature_extractor_cnn_1_6972177#
feature_extractor_cnn_1_6972179#
feature_extractor_cnn_1_6972181#
feature_extractor_cnn_1_6972183#
feature_extractor_cnn_1_6972185#
feature_extractor_cnn_1_6972187#
feature_extractor_cnn_1_6972189#
feature_extractor_cnn_1_6972191#
feature_extractor_cnn_1_6972193#
feature_extractor_cnn_1_6972195#
feature_extractor_cnn_1_6972197#
feature_extractor_cnn_1_6972199#
feature_extractor_cnn_1_6972201
dense_2_6972204
identity¢dense_2/StatefulPartitionedCall¢/feature_extractor_cnn_1/StatefulPartitionedCall[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/ys
truedivRealDivinputstruediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truedivë
/feature_extractor_cnn_1/StatefulPartitionedCallStatefulPartitionedCalltruediv:z:0feature_extractor_cnn_1_6972119feature_extractor_cnn_1_6972121feature_extractor_cnn_1_6972123feature_extractor_cnn_1_6972125feature_extractor_cnn_1_6972127feature_extractor_cnn_1_6972129feature_extractor_cnn_1_6972131feature_extractor_cnn_1_6972133feature_extractor_cnn_1_6972135feature_extractor_cnn_1_6972137feature_extractor_cnn_1_6972139feature_extractor_cnn_1_6972141feature_extractor_cnn_1_6972143feature_extractor_cnn_1_6972145feature_extractor_cnn_1_6972147feature_extractor_cnn_1_6972149feature_extractor_cnn_1_6972151feature_extractor_cnn_1_6972153feature_extractor_cnn_1_6972155feature_extractor_cnn_1_6972157feature_extractor_cnn_1_6972159feature_extractor_cnn_1_6972161feature_extractor_cnn_1_6972163feature_extractor_cnn_1_6972165feature_extractor_cnn_1_6972167feature_extractor_cnn_1_6972169feature_extractor_cnn_1_6972171feature_extractor_cnn_1_6972173feature_extractor_cnn_1_6972175feature_extractor_cnn_1_6972177feature_extractor_cnn_1_6972179feature_extractor_cnn_1_6972181feature_extractor_cnn_1_6972183feature_extractor_cnn_1_6972185feature_extractor_cnn_1_6972187feature_extractor_cnn_1_6972189feature_extractor_cnn_1_6972191feature_extractor_cnn_1_6972193feature_extractor_cnn_1_6972195feature_extractor_cnn_1_6972197feature_extractor_cnn_1_6972199feature_extractor_cnn_1_6972201*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**8
config_proto(&

CPU

GPU2*0J

   E8 *]
fXRV
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_697145621
/feature_extractor_cnn_1/StatefulPartitionedCall¼
dense_2/StatefulPartitionedCallStatefulPartitionedCall8feature_extractor_cnn_1/StatefulPartitionedCall:output:0dense_2_6972204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_69718192!
dense_2/StatefulPartitionedCallÐ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall0^feature_extractor_cnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2b
/feature_extractor_cnn_1/StatefulPartitionedCall/feature_extractor_cnn_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ7

T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6971271

inputs
cnn_block_0_6971178
cnn_block_0_6971180
cnn_block_0_6971182
cnn_block_0_6971184
cnn_block_0_6971186
cnn_block_0_6971188
cnn_block_1_6971191
cnn_block_1_6971193
cnn_block_1_6971195
cnn_block_1_6971197
cnn_block_1_6971199
cnn_block_1_6971201
cnn_block_2_6971204
cnn_block_2_6971206
cnn_block_2_6971208
cnn_block_2_6971210
cnn_block_2_6971212
cnn_block_2_6971214
cnn_block_3_6971217
cnn_block_3_6971219
cnn_block_3_6971221
cnn_block_3_6971223
cnn_block_3_6971225
cnn_block_3_6971227
cnn_block_4_6971230
cnn_block_4_6971232
cnn_block_4_6971234
cnn_block_4_6971236
cnn_block_4_6971238
cnn_block_4_6971240
cnn_block_5_6971243
cnn_block_5_6971245
cnn_block_5_6971247
cnn_block_5_6971249
cnn_block_5_6971251
cnn_block_5_6971253
cnn_block_6_6971256
cnn_block_6_6971258
cnn_block_6_6971260
cnn_block_6_6971262
cnn_block_6_6971264
cnn_block_6_6971266
identity¢#cnn_block_0/StatefulPartitionedCall¢#cnn_block_1/StatefulPartitionedCall¢#cnn_block_2/StatefulPartitionedCall¢#cnn_block_3/StatefulPartitionedCall¢#cnn_block_4/StatefulPartitionedCall¢#cnn_block_5/StatefulPartitionedCall¢#cnn_block_6/StatefulPartitionedCall
#cnn_block_0/StatefulPartitionedCallStatefulPartitionedCallinputscnn_block_0_6971178cnn_block_0_6971180cnn_block_0_6971182cnn_block_0_6971184cnn_block_0_6971186cnn_block_0_6971188*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69687992%
#cnn_block_0/StatefulPartitionedCall¹
#cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_0/StatefulPartitionedCall:output:0cnn_block_1_6971191cnn_block_1_6971193cnn_block_1_6971195cnn_block_1_6971197cnn_block_1_6971199cnn_block_1_6971201*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69691122%
#cnn_block_1/StatefulPartitionedCall¹
#cnn_block_2/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_1/StatefulPartitionedCall:output:0cnn_block_2_6971204cnn_block_2_6971206cnn_block_2_6971208cnn_block_2_6971210cnn_block_2_6971212cnn_block_2_6971214*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69694252%
#cnn_block_2/StatefulPartitionedCall¹
#cnn_block_3/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_2/StatefulPartitionedCall:output:0cnn_block_3_6971217cnn_block_3_6971219cnn_block_3_6971221cnn_block_3_6971223cnn_block_3_6971225cnn_block_3_6971227*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69697382%
#cnn_block_3/StatefulPartitionedCall¹
#cnn_block_4/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_3/StatefulPartitionedCall:output:0cnn_block_4_6971230cnn_block_4_6971232cnn_block_4_6971234cnn_block_4_6971236cnn_block_4_6971238cnn_block_4_6971240*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69700512%
#cnn_block_4/StatefulPartitionedCall¹
#cnn_block_5/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_4/StatefulPartitionedCall:output:0cnn_block_5_6971243cnn_block_5_6971245cnn_block_5_6971247cnn_block_5_6971249cnn_block_5_6971251cnn_block_5_6971253*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69703642%
#cnn_block_5/StatefulPartitionedCallº
#cnn_block_6/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_5/StatefulPartitionedCall:output:0cnn_block_6_6971256cnn_block_6_6971258cnn_block_6_6971260cnn_block_6_6971262cnn_block_6_6971264cnn_block_6_6971266*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69706772%
#cnn_block_6/StatefulPartitionedCall½
*global_average_pooling2d_1/PartitionedCallPartitionedCall,cnn_block_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *`
f[RY
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69707352,
*global_average_pooling2d_1/PartitionedCall
IdentityIdentity3global_average_pooling2d_1/PartitionedCall:output:0$^cnn_block_0/StatefulPartitionedCall$^cnn_block_1/StatefulPartitionedCall$^cnn_block_2/StatefulPartitionedCall$^cnn_block_3/StatefulPartitionedCall$^cnn_block_4/StatefulPartitionedCall$^cnn_block_5/StatefulPartitionedCall$^cnn_block_6/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::2J
#cnn_block_0/StatefulPartitionedCall#cnn_block_0/StatefulPartitionedCall2J
#cnn_block_1/StatefulPartitionedCall#cnn_block_1/StatefulPartitionedCall2J
#cnn_block_2/StatefulPartitionedCall#cnn_block_2/StatefulPartitionedCall2J
#cnn_block_3/StatefulPartitionedCall#cnn_block_3/StatefulPartitionedCall2J
#cnn_block_4/StatefulPartitionedCall#cnn_block_4/StatefulPartitionedCall2J
#cnn_block_5/StatefulPartitionedCall#cnn_block_5/StatefulPartitionedCall2J
#cnn_block_6/StatefulPartitionedCall#cnn_block_6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
-__inference_cnn_block_5_layer_call_fn_6975353

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69703642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò

S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6970273

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
í

H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974623

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp¿
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_8/BiasAdd¶
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp¼
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1é
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_8/FusedBatchNormV3
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1
re_lu_8/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_8/ReluÆ
IdentityIdentityre_lu_8/Relu:activations:0%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
ò
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6969148

inputs
conv2d_8_6969132
conv2d_8_6969134!
batch_normalization_8_6969137!
batch_normalization_8_6969139!
batch_normalization_8_6969141!
batch_normalization_8_6969143
identity¢-batch_normalization_8/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCallª
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_6969132conv2d_8_6969134*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_69689682"
 conv2d_8/StatefulPartitionedCallÐ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_6969137batch_normalization_8_6969139batch_normalization_8_6969141batch_normalization_8_6969143*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69690212/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_69690622
re_lu_8/PartitionedCallÏ
IdentityIdentity re_lu_8/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
«
8__inference_batch_normalization_11_layer_call_fn_6976339

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_69698822
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Å
ò
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6969461

inputs
conv2d_9_6969445
conv2d_9_6969447!
batch_normalization_9_6969450!
batch_normalization_9_6969452!
batch_normalization_9_6969454!
batch_normalization_9_6969456
identity¢-batch_normalization_9/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCallª
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_6969445conv2d_9_6969447*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_69692812"
 conv2d_9/StatefulPartitionedCallÐ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_9_6969450batch_normalization_9_6969452batch_normalization_9_6969454batch_normalization_9_6969456*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69693342/
-batch_normalization_9/StatefulPartitionedCall
re_lu_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_9_layer_call_and_return_conditional_losses_69693752
re_lu_9/PartitionedCallÏ
IdentityIdentity re_lu_9/PartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6969851

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§

S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976691

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ 

H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975397
conv2d_12_input,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource
identity¢%batch_normalization_12/AssignNewValue¢'batch_normalization_12/AssignNewValue_1³
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_12/Conv2D/ReadVariableOpË
conv2d_12/Conv2DConv2Dconv2d_12_input'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_12/Conv2Dª
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp°
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_12/BiasAdd¹
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOp¿
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1ì
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ö
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_12/FusedBatchNormV3
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValue
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1
re_lu_12/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_12/ReluÉ
IdentityIdentityre_lu_12/Relu:activations:0&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_1:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameconv2d_12_input


D__inference_dense_2_layer_call_and_return_conditional_losses_6971819

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
ú
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6970051

inputs
conv2d_11_6970035
conv2d_11_6970037"
batch_normalization_11_6970040"
batch_normalization_11_6970042"
batch_normalization_11_6970044"
batch_normalization_11_6970046
identity¢.batch_normalization_11/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¯
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_6970035conv2d_11_6970037*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_69699072#
!conv2d_11/StatefulPartitionedCallÖ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_6970040batch_normalization_11_6970042batch_normalization_11_6970044batch_normalization_11_6970046*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_696994220
.batch_normalization_11/StatefulPartitionedCall
re_lu_11/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_11_layer_call_and_return_conditional_losses_69700012
re_lu_11/PartitionedCallÒ
IdentityIdentity!re_lu_11/PartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ã7

T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6971456

inputs
cnn_block_0_6971363
cnn_block_0_6971365
cnn_block_0_6971367
cnn_block_0_6971369
cnn_block_0_6971371
cnn_block_0_6971373
cnn_block_1_6971376
cnn_block_1_6971378
cnn_block_1_6971380
cnn_block_1_6971382
cnn_block_1_6971384
cnn_block_1_6971386
cnn_block_2_6971389
cnn_block_2_6971391
cnn_block_2_6971393
cnn_block_2_6971395
cnn_block_2_6971397
cnn_block_2_6971399
cnn_block_3_6971402
cnn_block_3_6971404
cnn_block_3_6971406
cnn_block_3_6971408
cnn_block_3_6971410
cnn_block_3_6971412
cnn_block_4_6971415
cnn_block_4_6971417
cnn_block_4_6971419
cnn_block_4_6971421
cnn_block_4_6971423
cnn_block_4_6971425
cnn_block_5_6971428
cnn_block_5_6971430
cnn_block_5_6971432
cnn_block_5_6971434
cnn_block_5_6971436
cnn_block_5_6971438
cnn_block_6_6971441
cnn_block_6_6971443
cnn_block_6_6971445
cnn_block_6_6971447
cnn_block_6_6971449
cnn_block_6_6971451
identity¢#cnn_block_0/StatefulPartitionedCall¢#cnn_block_1/StatefulPartitionedCall¢#cnn_block_2/StatefulPartitionedCall¢#cnn_block_3/StatefulPartitionedCall¢#cnn_block_4/StatefulPartitionedCall¢#cnn_block_5/StatefulPartitionedCall¢#cnn_block_6/StatefulPartitionedCall
#cnn_block_0/StatefulPartitionedCallStatefulPartitionedCallinputscnn_block_0_6971363cnn_block_0_6971365cnn_block_0_6971367cnn_block_0_6971369cnn_block_0_6971371cnn_block_0_6971373*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69688352%
#cnn_block_0/StatefulPartitionedCall»
#cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_0/StatefulPartitionedCall:output:0cnn_block_1_6971376cnn_block_1_6971378cnn_block_1_6971380cnn_block_1_6971382cnn_block_1_6971384cnn_block_1_6971386*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69691482%
#cnn_block_1/StatefulPartitionedCall»
#cnn_block_2/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_1/StatefulPartitionedCall:output:0cnn_block_2_6971389cnn_block_2_6971391cnn_block_2_6971393cnn_block_2_6971395cnn_block_2_6971397cnn_block_2_6971399*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69694612%
#cnn_block_2/StatefulPartitionedCall»
#cnn_block_3/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_2/StatefulPartitionedCall:output:0cnn_block_3_6971402cnn_block_3_6971404cnn_block_3_6971406cnn_block_3_6971408cnn_block_3_6971410cnn_block_3_6971412*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69697742%
#cnn_block_3/StatefulPartitionedCall»
#cnn_block_4/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_3/StatefulPartitionedCall:output:0cnn_block_4_6971415cnn_block_4_6971417cnn_block_4_6971419cnn_block_4_6971421cnn_block_4_6971423cnn_block_4_6971425*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69700872%
#cnn_block_4/StatefulPartitionedCall»
#cnn_block_5/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_4/StatefulPartitionedCall:output:0cnn_block_5_6971428cnn_block_5_6971430cnn_block_5_6971432cnn_block_5_6971434cnn_block_5_6971436cnn_block_5_6971438*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69704002%
#cnn_block_5/StatefulPartitionedCall¼
#cnn_block_6/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_5/StatefulPartitionedCall:output:0cnn_block_6_6971441cnn_block_6_6971443cnn_block_6_6971445cnn_block_6_6971447cnn_block_6_6971449cnn_block_6_6971451*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69707132%
#cnn_block_6/StatefulPartitionedCall½
*global_average_pooling2d_1/PartitionedCallPartitionedCall,cnn_block_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *`
f[RY
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69707352,
*global_average_pooling2d_1/PartitionedCall
IdentityIdentity3global_average_pooling2d_1/PartitionedCall:output:0$^cnn_block_0/StatefulPartitionedCall$^cnn_block_1/StatefulPartitionedCall$^cnn_block_2/StatefulPartitionedCall$^cnn_block_3/StatefulPartitionedCall$^cnn_block_4/StatefulPartitionedCall$^cnn_block_5/StatefulPartitionedCall$^cnn_block_6/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::2J
#cnn_block_0/StatefulPartitionedCall#cnn_block_0/StatefulPartitionedCall2J
#cnn_block_1/StatefulPartitionedCall#cnn_block_1/StatefulPartitionedCall2J
#cnn_block_2/StatefulPartitionedCall#cnn_block_2/StatefulPartitionedCall2J
#cnn_block_3/StatefulPartitionedCall#cnn_block_3/StatefulPartitionedCall2J
#cnn_block_4/StatefulPartitionedCall#cnn_block_4/StatefulPartitionedCall2J
#cnn_block_5/StatefulPartitionedCall#cnn_block_5/StatefulPartitionedCall2J
#cnn_block_6/StatefulPartitionedCall#cnn_block_6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
Æ
-__inference_cnn_block_1_layer_call_fn_6974768
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69691482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_8_input

°
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976202

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
a
E__inference_re_lu_10_layer_call_and_return_conditional_losses_6969688

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
æ
«
8__inference_batch_normalization_10_layer_call_fn_6976246

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_69696472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
½
s
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_6970735

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ 

H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6975053
conv2d_10_input,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource
identity¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1³
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_10/Conv2D/ReadVariableOpË
conv2d_10/Conv2DConv2Dconv2d_10_input'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_10/Conv2Dª
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp°
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_10/BiasAdd¹
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp¿
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1ì
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ö
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/BiasAdd:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_10/FusedBatchNormV3
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1
re_lu_10/ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_10/ReluÉ
IdentityIdentityre_lu_10/Relu:activations:0&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_1:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)
_user_specified_nameconv2d_10_input


*__inference_conv2d_8_layer_call_fn_6975804

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_69689682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974451

inputs+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_7/AssignNewValue¢&batch_normalization_7/AssignNewValue_1°
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOp¿
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_7/Conv2D§
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp¬
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_7/BiasAdd¶
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp¼
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1é
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_7/FusedBatchNormV3
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1
re_lu_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_7/ReluÆ
IdentityIdentityre_lu_7/Relu:activations:0%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
E
)__inference_re_lu_7_layer_call_fn_6975785

inputs
identityÕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_69687492
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó
a
E__inference_re_lu_10_layer_call_and_return_conditional_losses_6976251

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ã
F
*__inference_re_lu_13_layer_call_fn_6976727

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_13_layer_call_and_return_conditional_losses_69706272
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
ª
7__inference_batch_normalization_7_layer_call_fn_6975698

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69685992
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
°
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6970477

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
«
8__inference_batch_normalization_13_layer_call_fn_6976653

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_69705862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6969942

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
«
Ç
-__inference_cnn_block_3_layer_call_fn_6975112
conv2d_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69697742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)
_user_specified_nameconv2d_10_input
É
Ã
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975422
conv2d_12_input,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource
identity³
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_12/Conv2D/ReadVariableOpË
conv2d_12/Conv2DConv2Dconv2d_12_input'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_12/Conv2Dª
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp°
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_12/BiasAdd¹
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOp¿
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1ì
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1è
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3
re_lu_12/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_12/Reluw
IdentityIdentityre_lu_12/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@:::::::` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameconv2d_12_input
Ê
¯
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975667

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÏÒ
ã'
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6972562
input_1O
Kfeature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceO
Kfeature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceO
Kfeature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource
identity¢Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue¢Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1¢Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue¢Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1¢Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue¢Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1¢Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue¢Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1¢Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue¢Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1¢Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue¢Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1¢Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue¢Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/yt
truedivRealDivinput_1truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv
Bfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02D
Bfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp°
3feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2DConv2Dtruediv:z:0Jfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D
Cfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02J
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02[
Yfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02]
[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ë
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3Û
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValueAssignVariableOpbfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceWfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:batch_mean:0Z^feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*u
_classk
igloc:@feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValueé
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1AssignVariableOpdfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:batch_variance:0\^feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*w
_classm
kiloc:@feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1ö
0feature_extractor_cnn_1/cnn_block_0/re_lu_7/ReluReluNfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu
Bfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02D
Bfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpã
3feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu:activations:0Jfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D
Cfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ë
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3Û
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValueAssignVariableOpbfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceWfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:batch_mean:0Z^feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*u
_classk
igloc:@feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValueé
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1AssignVariableOpdfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:batch_variance:0\^feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*w
_classm
kiloc:@feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1ö
0feature_extractor_cnn_1/cnn_block_1/re_lu_8/ReluReluNfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu
Bfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02D
Bfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpã
3feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu:activations:0Jfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D
Cfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ë
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3Û
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValueAssignVariableOpbfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceWfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:batch_mean:0Z^feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*u
_classk
igloc:@feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValueé
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1AssignVariableOpdfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:batch_variance:0\^feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*w
_classm
kiloc:@feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1ö
0feature_extractor_cnn_1/cnn_block_2/re_lu_9/ReluReluNfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu
Cfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpæ
4feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D
Dfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02F
Dfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 27
5feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02\
Zfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02^
\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ò
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3á
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValueAssignVariableOpcfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceXfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:batch_mean:0[^feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp*v
_classl
jhloc:@feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValueï
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1AssignVariableOpefeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:batch_variance:0]^feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*x
_classn
ljloc:@feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1ù
1feature_extractor_cnn_1/cnn_block_3/re_lu_10/ReluReluOfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu
Cfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpç
4feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D
Dfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02^
\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ò
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3á
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValueAssignVariableOpcfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceXfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:batch_mean:0[^feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp*v
_classl
jhloc:@feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValueï
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1AssignVariableOpefeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:batch_variance:0]^feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*x
_classn
ljloc:@feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1ù
1feature_extractor_cnn_1/cnn_block_4/re_lu_11/ReluReluOfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu
Cfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpç
4feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D
Dfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02^
\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ò
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3á
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValueAssignVariableOpcfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceXfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:batch_mean:0[^feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp*v
_classl
jhloc:@feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValueï
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1AssignVariableOpefeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:batch_variance:0]^feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*x
_classn
ljloc:@feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1ù
1feature_extractor_cnn_1/cnn_block_5/re_lu_12/ReluReluOfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu 
Cfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpè
4feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D
Dfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpÁ
5feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd¦
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp¬
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1Ù
Zfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpß
\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02^
\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1÷
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3á
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValueAssignVariableOpcfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceXfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:batch_mean:0[^feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp*v
_classl
jhloc:@feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValueï
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1AssignVariableOpefeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:batch_variance:0]^feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*x
_classn
ljloc:@feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1ú
1feature_extractor_cnn_1/cnn_block_6/re_lu_13/ReluReluOfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1feature_extractor_cnn_1/cnn_block_6/re_lu_13/Reluç
Ifeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2K
Ifeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesÂ
7feature_extractor_cnn_1/global_average_pooling2d_1/MeanMean?feature_extractor_cnn_1/cnn_block_6/re_lu_13/Relu:activations:0Rfeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7feature_extractor_cnn_1/global_average_pooling2d_1/Mean¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_2/MatMul/ReadVariableOpÅ
dense_2/MatMulMatMul@feature_extractor_cnn_1/global_average_pooling2d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_2/MatMul	
IdentityIdentitydense_2/MatMul:product:0I^feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValueK^feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1I^feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValueK^feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1I^feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValueK^feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1J^feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValueL^feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1J^feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValueL^feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1J^feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValueL^feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1J^feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValueL^feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::2
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValueHfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue2
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_12
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValueHfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue2
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_12
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValueHfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue2
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_12
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValueIfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue2
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_12
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValueIfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue2
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_12
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValueIfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue2
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_12
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValueIfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue2
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¨
®
F__inference_conv2d_12_layer_call_and_return_conditional_losses_6970220

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¾
-__inference_cnn_block_1_layer_call_fn_6974682

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69691482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
-__inference_cnn_block_3_layer_call_fn_6975009

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69697382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6970164

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
®
F__inference_conv2d_10_layer_call_and_return_conditional_losses_6976109

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
èì
³
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6972720
input_1O
Kfeature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceO
Kfeature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceO
Kfeature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/yt
truedivRealDivinput_1truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv
Bfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02D
Bfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp°
3feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2DConv2Dtruediv:z:0Jfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D
Cfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02J
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02[
Yfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02]
[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ý
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3ö
0feature_extractor_cnn_1/cnn_block_0/re_lu_7/ReluReluNfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu
Bfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02D
Bfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpã
3feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu:activations:0Jfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D
Cfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ý
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3ö
0feature_extractor_cnn_1/cnn_block_1/re_lu_8/ReluReluNfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu
Bfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02D
Bfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpã
3feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu:activations:0Jfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D
Cfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ý
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3ö
0feature_extractor_cnn_1/cnn_block_2/re_lu_9/ReluReluNfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu
Cfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpæ
4feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D
Dfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02F
Dfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 27
5feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02\
Zfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02^
\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ä
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3ù
1feature_extractor_cnn_1/cnn_block_3/re_lu_10/ReluReluOfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu
Cfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpç
4feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D
Dfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02^
\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ä
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3ù
1feature_extractor_cnn_1/cnn_block_4/re_lu_11/ReluReluOfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu
Cfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpç
4feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D
Dfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02^
\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ä
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3ù
1feature_extractor_cnn_1/cnn_block_5/re_lu_12/ReluReluOfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu 
Cfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpè
4feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D
Dfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpÁ
5feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd¦
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp¬
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1Ù
Zfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpß
\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02^
\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1é
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3ú
1feature_extractor_cnn_1/cnn_block_6/re_lu_13/ReluReluOfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1feature_extractor_cnn_1/cnn_block_6/re_lu_13/Reluç
Ifeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2K
Ifeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesÂ
7feature_extractor_cnn_1/global_average_pooling2d_1/MeanMean?feature_extractor_cnn_1/cnn_block_6/re_lu_13/Relu:activations:0Rfeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7feature_extractor_cnn_1/global_average_pooling2d_1/Mean¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_2/MatMul/ReadVariableOpÅ
dense_2/MatMulMatMul@feature_extractor_cnn_1/global_average_pooling2d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_2/MatMull
IdentityIdentitydense_2/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
°

T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6974232
input_17
3cnn_block_0_conv2d_7_conv2d_readvariableop_resource8
4cnn_block_0_conv2d_7_biasadd_readvariableop_resource=
9cnn_block_0_batch_normalization_7_readvariableop_resource?
;cnn_block_0_batch_normalization_7_readvariableop_1_resourceN
Jcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_1_conv2d_8_conv2d_readvariableop_resource8
4cnn_block_1_conv2d_8_biasadd_readvariableop_resource=
9cnn_block_1_batch_normalization_8_readvariableop_resource?
;cnn_block_1_batch_normalization_8_readvariableop_1_resourceN
Jcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_2_conv2d_9_conv2d_readvariableop_resource8
4cnn_block_2_conv2d_9_biasadd_readvariableop_resource=
9cnn_block_2_batch_normalization_9_readvariableop_resource?
;cnn_block_2_batch_normalization_9_readvariableop_1_resourceN
Jcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_3_conv2d_10_conv2d_readvariableop_resource9
5cnn_block_3_conv2d_10_biasadd_readvariableop_resource>
:cnn_block_3_batch_normalization_10_readvariableop_resource@
<cnn_block_3_batch_normalization_10_readvariableop_1_resourceO
Kcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_4_conv2d_11_conv2d_readvariableop_resource9
5cnn_block_4_conv2d_11_biasadd_readvariableop_resource>
:cnn_block_4_batch_normalization_11_readvariableop_resource@
<cnn_block_4_batch_normalization_11_readvariableop_1_resourceO
Kcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_5_conv2d_12_conv2d_readvariableop_resource9
5cnn_block_5_conv2d_12_biasadd_readvariableop_resource>
:cnn_block_5_batch_normalization_12_readvariableop_resource@
<cnn_block_5_batch_normalization_12_readvariableop_1_resourceO
Kcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_6_conv2d_13_conv2d_readvariableop_resource9
5cnn_block_6_conv2d_13_biasadd_readvariableop_resource>
:cnn_block_6_batch_normalization_13_readvariableop_resource@
<cnn_block_6_batch_normalization_13_readvariableop_1_resourceO
Kcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource
identityÔ
*cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*cnn_block_0/conv2d_7/Conv2D/ReadVariableOpä
cnn_block_0/conv2d_7/Conv2DConv2Dinput_12cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_0/conv2d_7/Conv2DË
+cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpÜ
cnn_block_0/conv2d_7/BiasAddBiasAdd$cnn_block_0/conv2d_7/Conv2D:output:03cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/conv2d_7/BiasAddÚ
0cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype022
0cnn_block_0/batch_normalization_7/ReadVariableOpà
2cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype024
2cnn_block_0/batch_normalization_7/ReadVariableOp_1
Acnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02C
Acnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
Ccnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02E
Ccnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_0/conv2d_7/BiasAdd:output:08cnn_block_0/batch_normalization_7/ReadVariableOp:value:0:cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 24
2cnn_block_0/batch_normalization_7/FusedBatchNormV3®
cnn_block_0/re_lu_7/ReluRelu6cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/re_lu_7/ReluÔ
*cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_1/conv2d_8/Conv2D/ReadVariableOp
cnn_block_1/conv2d_8/Conv2DConv2D&cnn_block_0/re_lu_7/Relu:activations:02cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_1/conv2d_8/Conv2DË
+cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpÜ
cnn_block_1/conv2d_8/BiasAddBiasAdd$cnn_block_1/conv2d_8/Conv2D:output:03cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/conv2d_8/BiasAddÚ
0cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_1/batch_normalization_8/ReadVariableOpà
2cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_1/batch_normalization_8/ReadVariableOp_1
Acnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
Ccnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_1/conv2d_8/BiasAdd:output:08cnn_block_1/batch_normalization_8/ReadVariableOp:value:0:cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_1/batch_normalization_8/FusedBatchNormV3®
cnn_block_1/re_lu_8/ReluRelu6cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/re_lu_8/ReluÔ
*cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_2/conv2d_9/Conv2D/ReadVariableOp
cnn_block_2/conv2d_9/Conv2DConv2D&cnn_block_1/re_lu_8/Relu:activations:02cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_2/conv2d_9/Conv2DË
+cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpÜ
cnn_block_2/conv2d_9/BiasAddBiasAdd$cnn_block_2/conv2d_9/Conv2D:output:03cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/conv2d_9/BiasAddÚ
0cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOp9cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_2/batch_normalization_9/ReadVariableOpà
2cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp;cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_2/batch_normalization_9/ReadVariableOp_1
Acnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
Ccnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%cnn_block_2/conv2d_9/BiasAdd:output:08cnn_block_2/batch_normalization_9/ReadVariableOp:value:0:cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0Icnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_2/batch_normalization_9/FusedBatchNormV3®
cnn_block_2/re_lu_9/ReluRelu6cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/re_lu_9/Relu×
+cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+cnn_block_3/conv2d_10/Conv2D/ReadVariableOp
cnn_block_3/conv2d_10/Conv2DConv2D&cnn_block_2/re_lu_9/Relu:activations:03cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_3/conv2d_10/Conv2DÎ
,cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpà
cnn_block_3/conv2d_10/BiasAddBiasAdd%cnn_block_3/conv2d_10/Conv2D:output:04cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/conv2d_10/BiasAddÝ
1cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOp:cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype023
1cnn_block_3/batch_normalization_10/ReadVariableOpã
3cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOp<cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype025
3cnn_block_3/batch_normalization_10/ReadVariableOp_1
Bcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
Dcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¼
3cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3&cnn_block_3/conv2d_10/BiasAdd:output:09cnn_block_3/batch_normalization_10/ReadVariableOp:value:0;cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0Jcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 25
3cnn_block_3/batch_normalization_10/FusedBatchNormV3±
cnn_block_3/re_lu_10/ReluRelu7cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/re_lu_10/Relu×
+cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+cnn_block_4/conv2d_11/Conv2D/ReadVariableOp
cnn_block_4/conv2d_11/Conv2DConv2D'cnn_block_3/re_lu_10/Relu:activations:03cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_4/conv2d_11/Conv2DÎ
,cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpà
cnn_block_4/conv2d_11/BiasAddBiasAdd%cnn_block_4/conv2d_11/Conv2D:output:04cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/conv2d_11/BiasAddÝ
1cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOp:cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype023
1cnn_block_4/batch_normalization_11/ReadVariableOpã
3cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOp<cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3cnn_block_4/batch_normalization_11/ReadVariableOp_1
Bcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
Dcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¼
3cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&cnn_block_4/conv2d_11/BiasAdd:output:09cnn_block_4/batch_normalization_11/ReadVariableOp:value:0;cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0Jcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 25
3cnn_block_4/batch_normalization_11/FusedBatchNormV3±
cnn_block_4/re_lu_11/ReluRelu7cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/re_lu_11/Relu×
+cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+cnn_block_5/conv2d_12/Conv2D/ReadVariableOp
cnn_block_5/conv2d_12/Conv2DConv2D'cnn_block_4/re_lu_11/Relu:activations:03cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_5/conv2d_12/Conv2DÎ
,cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpà
cnn_block_5/conv2d_12/BiasAddBiasAdd%cnn_block_5/conv2d_12/Conv2D:output:04cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/conv2d_12/BiasAddÝ
1cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOp:cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype023
1cnn_block_5/batch_normalization_12/ReadVariableOpã
3cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOp<cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3cnn_block_5/batch_normalization_12/ReadVariableOp_1
Bcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp
Dcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1¼
3cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3&cnn_block_5/conv2d_12/BiasAdd:output:09cnn_block_5/batch_normalization_12/ReadVariableOp:value:0;cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0Jcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 25
3cnn_block_5/batch_normalization_12/FusedBatchNormV3±
cnn_block_5/re_lu_12/ReluRelu7cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/re_lu_12/ReluØ
+cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+cnn_block_6/conv2d_13/Conv2D/ReadVariableOp
cnn_block_6/conv2d_13/Conv2DConv2D'cnn_block_5/re_lu_12/Relu:activations:03cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_6/conv2d_13/Conv2DÏ
,cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpá
cnn_block_6/conv2d_13/BiasAddBiasAdd%cnn_block_6/conv2d_13/Conv2D:output:04cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/conv2d_13/BiasAddÞ
1cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOp:cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype023
1cnn_block_6/batch_normalization_13/ReadVariableOpä
3cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOp<cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3cnn_block_6/batch_normalization_13/ReadVariableOp_1
Bcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp
Dcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Á
3cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3&cnn_block_6/conv2d_13/BiasAdd:output:09cnn_block_6/batch_normalization_13/ReadVariableOp:value:0;cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0Jcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 25
3cnn_block_6/batch_normalization_13/FusedBatchNormV3²
cnn_block_6/re_lu_13/ReluRelu7cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/re_lu_13/Relu·
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indicesâ
global_average_pooling2d_1/MeanMean'cnn_block_6/re_lu_13/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
global_average_pooling2d_1/Mean}
IdentityIdentity(global_average_pooling2d_1/Mean:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

¾
-__inference_cnn_block_6_layer_call_fn_6975611

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69706772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò
`
D__inference_re_lu_9_layer_call_and_return_conditional_losses_6976094

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


+__inference_conv2d_11_layer_call_fn_6976275

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_69699072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
½
E
)__inference_re_lu_8_layer_call_fn_6975942

inputs
identityÕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_69690622
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
ª
7__inference_batch_normalization_8_layer_call_fn_6975919

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69690032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç
ú
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6970677

inputs
conv2d_13_6970661
conv2d_13_6970663"
batch_normalization_13_6970666"
batch_normalization_13_6970668"
batch_normalization_13_6970670"
batch_normalization_13_6970672
identity¢.batch_normalization_13/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall°
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_13_6970661conv2d_13_6970663*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_69705332#
!conv2d_13/StatefulPartitionedCall×
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_13_6970666batch_normalization_13_6970668batch_normalization_13_6970670batch_normalization_13_6970672*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_697056820
.batch_normalization_13/StatefulPartitionedCall
re_lu_13/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_13_layer_call_and_return_conditional_losses_69706272
re_lu_13/PartitionedCallÓ
IdentityIdentity!re_lu_13/PartitionedCall:output:0/^batch_normalization_13/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î

1__inference_contrastive_cnn_layer_call_fn_6972902
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*8
config_proto(&

CPU

GPU2*0J

   E8 *U
fPRN
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_69722082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
©
Ç
-__inference_cnn_block_4_layer_call_fn_6975267
conv2d_11_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallconv2d_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69700512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)
_user_specified_nameconv2d_11_input

¼
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974562
conv2d_7_input+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOpÇ
conv2d_7/Conv2DConv2Dconv2d_7_input&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_7/Conv2D§
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp¬
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_7/BiasAdd¶
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp¼
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1é
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
re_lu_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_7/Reluv
IdentityIdentityre_lu_7/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:::::::_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_7_input

°
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6969629

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6968630

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æV
Ï
 __inference__traced_save_6976879
file_prefix=
9savev2_contrastive_cnn_dense_2_kernel_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9e91203f12074d6eaf858195447470ea/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameù
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*
valueBþ,B&head/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesà
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_contrastive_cnn_dense_2_kernel_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ú
_input_shapesè
å: :	@::::: : : : :  : : : :  : : : : @:@:@:@:@@:@:@:@:@:::::: : : : : : :@:@:@:@::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	@:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: :,
(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@:!*

_output_shapes	
::!+

_output_shapes	
::,

_output_shapes
: 
Ê
¯
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975824

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975731

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
ª
7__inference_batch_normalization_8_layer_call_fn_6975855

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69689122
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í

H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974881

inputs+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_9/AssignNewValue¢&batch_normalization_9/AssignNewValue_1°
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_9/Conv2D/ReadVariableOp¿
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_9/Conv2D§
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp¬
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_9/BiasAdd¶
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp¼
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1é
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_9/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_9/FusedBatchNormV3
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1
re_lu_9/ReluRelu*batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_9/ReluÆ
IdentityIdentityre_lu_9/Relu:activations:0%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ã
ò
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6969112

inputs
conv2d_8_6969096
conv2d_8_6969098!
batch_normalization_8_6969101!
batch_normalization_8_6969103!
batch_normalization_8_6969105!
batch_normalization_8_6969107
identity¢-batch_normalization_8/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCallª
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_6969096conv2d_8_6969098*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_69689682"
 conv2d_8/StatefulPartitionedCallÎ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_6969101batch_normalization_8_6969103batch_normalization_8_6969105batch_normalization_8_6969107*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69690032/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_8_layer_call_and_return_conditional_losses_69690622
re_lu_8/PartitionedCallÏ
IdentityIdentity re_lu_8/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974537
conv2d_7_input+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_7/AssignNewValue¢&batch_normalization_7/AssignNewValue_1°
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOpÇ
conv2d_7/Conv2DConv2Dconv2d_7_input&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_7/Conv2D§
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp¬
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_7/BiasAdd¶
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp¼
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1é
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_7/FusedBatchNormV3
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1
re_lu_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_7/ReluÆ
IdentityIdentityre_lu_7/Relu:activations:0%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_7_input
ã
ú
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6969738

inputs
conv2d_10_6969722
conv2d_10_6969724"
batch_normalization_10_6969727"
batch_normalization_10_6969729"
batch_normalization_10_6969731"
batch_normalization_10_6969733
identity¢.batch_normalization_10/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¯
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_6969722conv2d_10_6969724*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_69695942#
!conv2d_10/StatefulPartitionedCallÖ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_6969727batch_normalization_10_6969729batch_normalization_10_6969731batch_normalization_10_6969733*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_696962920
.batch_normalization_10/StatefulPartitionedCall
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_69696882
re_lu_10/PartitionedCallÒ
IdentityIdentity!re_lu_10/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
½
E
)__inference_re_lu_9_layer_call_fn_6976099

inputs
identityÕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_9_layer_call_and_return_conditional_losses_69693752
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ò

S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976377

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


+__inference_conv2d_13_layer_call_fn_6976589

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_69705332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

°
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6970568

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
ª
7__inference_batch_normalization_9_layer_call_fn_6976076

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69692252
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¾
-__inference_cnn_block_4_layer_call_fn_6975198

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69700872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976156

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


+__inference_conv2d_10_layer_call_fn_6976118

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_69695942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


+__inference_conv2d_12_layer_call_fn_6976432

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_69702202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
å
ú
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6970087

inputs
conv2d_11_6970071
conv2d_11_6970073"
batch_normalization_11_6970076"
batch_normalization_11_6970078"
batch_normalization_11_6970080"
batch_normalization_11_6970082
identity¢.batch_normalization_11/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¯
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_6970071conv2d_11_6970073*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_11_layer_call_and_return_conditional_losses_69699072#
!conv2d_11/StatefulPartitionedCallØ
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_6970076batch_normalization_11_6970078batch_normalization_11_6970080batch_normalization_11_6970082*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_696996020
.batch_normalization_11/StatefulPartitionedCall
re_lu_11/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_11_layer_call_and_return_conditional_losses_69700012
re_lu_11/PartitionedCallÒ
IdentityIdentity!re_lu_11/PartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ã
ú
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6970364

inputs
conv2d_12_6970348
conv2d_12_6970350"
batch_normalization_12_6970353"
batch_normalization_12_6970355"
batch_normalization_12_6970357"
batch_normalization_12_6970359
identity¢.batch_normalization_12/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¯
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_6970348conv2d_12_6970350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_69702202#
!conv2d_12/StatefulPartitionedCallÖ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_12_6970353batch_normalization_12_6970355batch_normalization_12_6970357batch_normalization_12_6970359*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_697025520
.batch_normalization_12/StatefulPartitionedCall
re_lu_12/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_12_layer_call_and_return_conditional_losses_69703142
re_lu_12/PartitionedCallÒ
IdentityIdentity!re_lu_12/PartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975906

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

°
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976359

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6969334

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
«
8__inference_batch_normalization_10_layer_call_fn_6976182

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_69695692
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
à

1__inference_contrastive_cnn_layer_call_fn_6972811
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*?
_read_only_resource_inputs!
	
 !"%&'(+*8
config_proto(&

CPU

GPU2*0J

   E8 *U
fPRN
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_69720232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ºû
ÿ
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6974079
input_17
3cnn_block_0_conv2d_7_conv2d_readvariableop_resource8
4cnn_block_0_conv2d_7_biasadd_readvariableop_resource=
9cnn_block_0_batch_normalization_7_readvariableop_resource?
;cnn_block_0_batch_normalization_7_readvariableop_1_resourceN
Jcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_1_conv2d_8_conv2d_readvariableop_resource8
4cnn_block_1_conv2d_8_biasadd_readvariableop_resource=
9cnn_block_1_batch_normalization_8_readvariableop_resource?
;cnn_block_1_batch_normalization_8_readvariableop_1_resourceN
Jcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_2_conv2d_9_conv2d_readvariableop_resource8
4cnn_block_2_conv2d_9_biasadd_readvariableop_resource=
9cnn_block_2_batch_normalization_9_readvariableop_resource?
;cnn_block_2_batch_normalization_9_readvariableop_1_resourceN
Jcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_3_conv2d_10_conv2d_readvariableop_resource9
5cnn_block_3_conv2d_10_biasadd_readvariableop_resource>
:cnn_block_3_batch_normalization_10_readvariableop_resource@
<cnn_block_3_batch_normalization_10_readvariableop_1_resourceO
Kcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_4_conv2d_11_conv2d_readvariableop_resource9
5cnn_block_4_conv2d_11_biasadd_readvariableop_resource>
:cnn_block_4_batch_normalization_11_readvariableop_resource@
<cnn_block_4_batch_normalization_11_readvariableop_1_resourceO
Kcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_5_conv2d_12_conv2d_readvariableop_resource9
5cnn_block_5_conv2d_12_biasadd_readvariableop_resource>
:cnn_block_5_batch_normalization_12_readvariableop_resource@
<cnn_block_5_batch_normalization_12_readvariableop_1_resourceO
Kcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_6_conv2d_13_conv2d_readvariableop_resource9
5cnn_block_6_conv2d_13_biasadd_readvariableop_resource>
:cnn_block_6_batch_normalization_13_readvariableop_resource@
<cnn_block_6_batch_normalization_13_readvariableop_1_resourceO
Kcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource
identity¢0cnn_block_0/batch_normalization_7/AssignNewValue¢2cnn_block_0/batch_normalization_7/AssignNewValue_1¢0cnn_block_1/batch_normalization_8/AssignNewValue¢2cnn_block_1/batch_normalization_8/AssignNewValue_1¢0cnn_block_2/batch_normalization_9/AssignNewValue¢2cnn_block_2/batch_normalization_9/AssignNewValue_1¢1cnn_block_3/batch_normalization_10/AssignNewValue¢3cnn_block_3/batch_normalization_10/AssignNewValue_1¢1cnn_block_4/batch_normalization_11/AssignNewValue¢3cnn_block_4/batch_normalization_11/AssignNewValue_1¢1cnn_block_5/batch_normalization_12/AssignNewValue¢3cnn_block_5/batch_normalization_12/AssignNewValue_1¢1cnn_block_6/batch_normalization_13/AssignNewValue¢3cnn_block_6/batch_normalization_13/AssignNewValue_1Ô
*cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*cnn_block_0/conv2d_7/Conv2D/ReadVariableOpä
cnn_block_0/conv2d_7/Conv2DConv2Dinput_12cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_0/conv2d_7/Conv2DË
+cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpÜ
cnn_block_0/conv2d_7/BiasAddBiasAdd$cnn_block_0/conv2d_7/Conv2D:output:03cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/conv2d_7/BiasAddÚ
0cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype022
0cnn_block_0/batch_normalization_7/ReadVariableOpà
2cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype024
2cnn_block_0/batch_normalization_7/ReadVariableOp_1
Acnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02C
Acnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
Ccnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02E
Ccnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_0/conv2d_7/BiasAdd:output:08cnn_block_0/batch_normalization_7/ReadVariableOp:value:0:cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_0/batch_normalization_7/FusedBatchNormV3Ë
0cnn_block_0/batch_normalization_7/AssignNewValueAssignVariableOpJcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource?cnn_block_0/batch_normalization_7/FusedBatchNormV3:batch_mean:0B^cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_0/batch_normalization_7/AssignNewValueÙ
2cnn_block_0/batch_normalization_7/AssignNewValue_1AssignVariableOpLcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_0/batch_normalization_7/FusedBatchNormV3:batch_variance:0D^cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_0/batch_normalization_7/AssignNewValue_1®
cnn_block_0/re_lu_7/ReluRelu6cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/re_lu_7/ReluÔ
*cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_1/conv2d_8/Conv2D/ReadVariableOp
cnn_block_1/conv2d_8/Conv2DConv2D&cnn_block_0/re_lu_7/Relu:activations:02cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_1/conv2d_8/Conv2DË
+cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpÜ
cnn_block_1/conv2d_8/BiasAddBiasAdd$cnn_block_1/conv2d_8/Conv2D:output:03cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/conv2d_8/BiasAddÚ
0cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_1/batch_normalization_8/ReadVariableOpà
2cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_1/batch_normalization_8/ReadVariableOp_1
Acnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
Ccnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_1/conv2d_8/BiasAdd:output:08cnn_block_1/batch_normalization_8/ReadVariableOp:value:0:cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_1/batch_normalization_8/FusedBatchNormV3Ë
0cnn_block_1/batch_normalization_8/AssignNewValueAssignVariableOpJcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource?cnn_block_1/batch_normalization_8/FusedBatchNormV3:batch_mean:0B^cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_1/batch_normalization_8/AssignNewValueÙ
2cnn_block_1/batch_normalization_8/AssignNewValue_1AssignVariableOpLcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_1/batch_normalization_8/FusedBatchNormV3:batch_variance:0D^cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_1/batch_normalization_8/AssignNewValue_1®
cnn_block_1/re_lu_8/ReluRelu6cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/re_lu_8/ReluÔ
*cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_2/conv2d_9/Conv2D/ReadVariableOp
cnn_block_2/conv2d_9/Conv2DConv2D&cnn_block_1/re_lu_8/Relu:activations:02cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_2/conv2d_9/Conv2DË
+cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpÜ
cnn_block_2/conv2d_9/BiasAddBiasAdd$cnn_block_2/conv2d_9/Conv2D:output:03cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/conv2d_9/BiasAddÚ
0cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOp9cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_2/batch_normalization_9/ReadVariableOpà
2cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp;cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_2/batch_normalization_9/ReadVariableOp_1
Acnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
Ccnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%cnn_block_2/conv2d_9/BiasAdd:output:08cnn_block_2/batch_normalization_9/ReadVariableOp:value:0:cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0Icnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_2/batch_normalization_9/FusedBatchNormV3Ë
0cnn_block_2/batch_normalization_9/AssignNewValueAssignVariableOpJcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource?cnn_block_2/batch_normalization_9/FusedBatchNormV3:batch_mean:0B^cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_2/batch_normalization_9/AssignNewValueÙ
2cnn_block_2/batch_normalization_9/AssignNewValue_1AssignVariableOpLcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_2/batch_normalization_9/FusedBatchNormV3:batch_variance:0D^cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_2/batch_normalization_9/AssignNewValue_1®
cnn_block_2/re_lu_9/ReluRelu6cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/re_lu_9/Relu×
+cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+cnn_block_3/conv2d_10/Conv2D/ReadVariableOp
cnn_block_3/conv2d_10/Conv2DConv2D&cnn_block_2/re_lu_9/Relu:activations:03cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_3/conv2d_10/Conv2DÎ
,cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpà
cnn_block_3/conv2d_10/BiasAddBiasAdd%cnn_block_3/conv2d_10/Conv2D:output:04cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/conv2d_10/BiasAddÝ
1cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOp:cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype023
1cnn_block_3/batch_normalization_10/ReadVariableOpã
3cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOp<cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype025
3cnn_block_3/batch_normalization_10/ReadVariableOp_1
Bcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
Dcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ê
3cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3&cnn_block_3/conv2d_10/BiasAdd:output:09cnn_block_3/batch_normalization_10/ReadVariableOp:value:0;cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0Jcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<25
3cnn_block_3/batch_normalization_10/FusedBatchNormV3Ñ
1cnn_block_3/batch_normalization_10/AssignNewValueAssignVariableOpKcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource@cnn_block_3/batch_normalization_10/FusedBatchNormV3:batch_mean:0C^cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1cnn_block_3/batch_normalization_10/AssignNewValueß
3cnn_block_3/batch_normalization_10/AssignNewValue_1AssignVariableOpMcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceDcnn_block_3/batch_normalization_10/FusedBatchNormV3:batch_variance:0E^cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3cnn_block_3/batch_normalization_10/AssignNewValue_1±
cnn_block_3/re_lu_10/ReluRelu7cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/re_lu_10/Relu×
+cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+cnn_block_4/conv2d_11/Conv2D/ReadVariableOp
cnn_block_4/conv2d_11/Conv2DConv2D'cnn_block_3/re_lu_10/Relu:activations:03cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_4/conv2d_11/Conv2DÎ
,cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpà
cnn_block_4/conv2d_11/BiasAddBiasAdd%cnn_block_4/conv2d_11/Conv2D:output:04cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/conv2d_11/BiasAddÝ
1cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOp:cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype023
1cnn_block_4/batch_normalization_11/ReadVariableOpã
3cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOp<cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3cnn_block_4/batch_normalization_11/ReadVariableOp_1
Bcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
Dcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ê
3cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&cnn_block_4/conv2d_11/BiasAdd:output:09cnn_block_4/batch_normalization_11/ReadVariableOp:value:0;cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0Jcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<25
3cnn_block_4/batch_normalization_11/FusedBatchNormV3Ñ
1cnn_block_4/batch_normalization_11/AssignNewValueAssignVariableOpKcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource@cnn_block_4/batch_normalization_11/FusedBatchNormV3:batch_mean:0C^cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1cnn_block_4/batch_normalization_11/AssignNewValueß
3cnn_block_4/batch_normalization_11/AssignNewValue_1AssignVariableOpMcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceDcnn_block_4/batch_normalization_11/FusedBatchNormV3:batch_variance:0E^cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3cnn_block_4/batch_normalization_11/AssignNewValue_1±
cnn_block_4/re_lu_11/ReluRelu7cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/re_lu_11/Relu×
+cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+cnn_block_5/conv2d_12/Conv2D/ReadVariableOp
cnn_block_5/conv2d_12/Conv2DConv2D'cnn_block_4/re_lu_11/Relu:activations:03cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_5/conv2d_12/Conv2DÎ
,cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpà
cnn_block_5/conv2d_12/BiasAddBiasAdd%cnn_block_5/conv2d_12/Conv2D:output:04cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/conv2d_12/BiasAddÝ
1cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOp:cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype023
1cnn_block_5/batch_normalization_12/ReadVariableOpã
3cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOp<cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3cnn_block_5/batch_normalization_12/ReadVariableOp_1
Bcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp
Dcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ê
3cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3&cnn_block_5/conv2d_12/BiasAdd:output:09cnn_block_5/batch_normalization_12/ReadVariableOp:value:0;cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0Jcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<25
3cnn_block_5/batch_normalization_12/FusedBatchNormV3Ñ
1cnn_block_5/batch_normalization_12/AssignNewValueAssignVariableOpKcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource@cnn_block_5/batch_normalization_12/FusedBatchNormV3:batch_mean:0C^cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1cnn_block_5/batch_normalization_12/AssignNewValueß
3cnn_block_5/batch_normalization_12/AssignNewValue_1AssignVariableOpMcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceDcnn_block_5/batch_normalization_12/FusedBatchNormV3:batch_variance:0E^cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3cnn_block_5/batch_normalization_12/AssignNewValue_1±
cnn_block_5/re_lu_12/ReluRelu7cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/re_lu_12/ReluØ
+cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+cnn_block_6/conv2d_13/Conv2D/ReadVariableOp
cnn_block_6/conv2d_13/Conv2DConv2D'cnn_block_5/re_lu_12/Relu:activations:03cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_6/conv2d_13/Conv2DÏ
,cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpá
cnn_block_6/conv2d_13/BiasAddBiasAdd%cnn_block_6/conv2d_13/Conv2D:output:04cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/conv2d_13/BiasAddÞ
1cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOp:cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype023
1cnn_block_6/batch_normalization_13/ReadVariableOpä
3cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOp<cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3cnn_block_6/batch_normalization_13/ReadVariableOp_1
Bcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp
Dcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Ï
3cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3&cnn_block_6/conv2d_13/BiasAdd:output:09cnn_block_6/batch_normalization_13/ReadVariableOp:value:0;cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0Jcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<25
3cnn_block_6/batch_normalization_13/FusedBatchNormV3Ñ
1cnn_block_6/batch_normalization_13/AssignNewValueAssignVariableOpKcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource@cnn_block_6/batch_normalization_13/FusedBatchNormV3:batch_mean:0C^cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1cnn_block_6/batch_normalization_13/AssignNewValueß
3cnn_block_6/batch_normalization_13/AssignNewValue_1AssignVariableOpMcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resourceDcnn_block_6/batch_normalization_13/FusedBatchNormV3:batch_variance:0E^cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3cnn_block_6/batch_normalization_13/AssignNewValue_1²
cnn_block_6/re_lu_13/ReluRelu7cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/re_lu_13/Relu·
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indicesâ
global_average_pooling2d_1/MeanMean'cnn_block_6/re_lu_13/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
global_average_pooling2d_1/MeanÝ
IdentityIdentity(global_average_pooling2d_1/Mean:output:01^cnn_block_0/batch_normalization_7/AssignNewValue3^cnn_block_0/batch_normalization_7/AssignNewValue_11^cnn_block_1/batch_normalization_8/AssignNewValue3^cnn_block_1/batch_normalization_8/AssignNewValue_11^cnn_block_2/batch_normalization_9/AssignNewValue3^cnn_block_2/batch_normalization_9/AssignNewValue_12^cnn_block_3/batch_normalization_10/AssignNewValue4^cnn_block_3/batch_normalization_10/AssignNewValue_12^cnn_block_4/batch_normalization_11/AssignNewValue4^cnn_block_4/batch_normalization_11/AssignNewValue_12^cnn_block_5/batch_normalization_12/AssignNewValue4^cnn_block_5/batch_normalization_12/AssignNewValue_12^cnn_block_6/batch_normalization_13/AssignNewValue4^cnn_block_6/batch_normalization_13/AssignNewValue_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::2d
0cnn_block_0/batch_normalization_7/AssignNewValue0cnn_block_0/batch_normalization_7/AssignNewValue2h
2cnn_block_0/batch_normalization_7/AssignNewValue_12cnn_block_0/batch_normalization_7/AssignNewValue_12d
0cnn_block_1/batch_normalization_8/AssignNewValue0cnn_block_1/batch_normalization_8/AssignNewValue2h
2cnn_block_1/batch_normalization_8/AssignNewValue_12cnn_block_1/batch_normalization_8/AssignNewValue_12d
0cnn_block_2/batch_normalization_9/AssignNewValue0cnn_block_2/batch_normalization_9/AssignNewValue2h
2cnn_block_2/batch_normalization_9/AssignNewValue_12cnn_block_2/batch_normalization_9/AssignNewValue_12f
1cnn_block_3/batch_normalization_10/AssignNewValue1cnn_block_3/batch_normalization_10/AssignNewValue2j
3cnn_block_3/batch_normalization_10/AssignNewValue_13cnn_block_3/batch_normalization_10/AssignNewValue_12f
1cnn_block_4/batch_normalization_11/AssignNewValue1cnn_block_4/batch_normalization_11/AssignNewValue2j
3cnn_block_4/batch_normalization_11/AssignNewValue_13cnn_block_4/batch_normalization_11/AssignNewValue_12f
1cnn_block_5/batch_normalization_12/AssignNewValue1cnn_block_5/batch_normalization_12/AssignNewValue2j
3cnn_block_5/batch_normalization_12/AssignNewValue_13cnn_block_5/batch_normalization_12/AssignNewValue_12f
1cnn_block_6/batch_normalization_13/AssignNewValue1cnn_block_6/batch_normalization_13/AssignNewValue2j
3cnn_block_6/batch_normalization_13/AssignNewValue_13cnn_block_6/batch_normalization_13/AssignNewValue_1:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975685

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
«
8__inference_batch_normalization_11_layer_call_fn_6976390

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_69699422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
×
a
E__inference_re_lu_13_layer_call_and_return_conditional_losses_6970627

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
«
8__inference_batch_normalization_10_layer_call_fn_6976233

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_69696292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6968943

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976295

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§
­
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6969281

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¾
-__inference_cnn_block_1_layer_call_fn_6974665

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69691122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
®
F__inference_conv2d_13_layer_call_and_return_conditional_losses_6970533

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
Æ
-__inference_cnn_block_2_layer_call_fn_6974854
conv2d_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69694612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_9_input


D__inference_dense_2_layer_call_and_return_conditional_losses_6974417

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6969647

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¦
Æ
-__inference_cnn_block_0_layer_call_fn_6974579
conv2d_7_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69687992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_7_input
§
­
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6975638

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6968708

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976313

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ó
a
E__inference_re_lu_12_layer_call_and_return_conditional_losses_6976565

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò

S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976534

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿
F
*__inference_re_lu_11_layer_call_fn_6976413

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_11_layer_call_and_return_conditional_losses_69700012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6976063

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
«
Ç
-__inference_cnn_block_4_layer_call_fn_6975284
conv2d_11_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallconv2d_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69700872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)
_user_specified_nameconv2d_11_input

°
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976516

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
« 

H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975139

inputs,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identity¢%batch_normalization_11/AssignNewValue¢'batch_normalization_11/AssignNewValue_1³
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_11/Conv2D/ReadVariableOpÂ
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_11/Conv2Dª
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp°
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_11/BiasAdd¹
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOp¿
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1ì
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ö
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_11/FusedBatchNormV3
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValue
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1
re_lu_11/ReluRelu+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_11/ReluÉ
IdentityIdentityre_lu_11/Relu:activations:0&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬
«
8__inference_batch_normalization_12_layer_call_fn_6976483

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_69701642
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6968912

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
a
E__inference_re_lu_11_layer_call_and_return_conditional_losses_6976408

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6975981

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§
­
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6975952

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¾
-__inference_cnn_block_0_layer_call_fn_6974493

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69687992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸

%__inference_signature_wrapper_6972390
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*8
config_proto(&

CPU

GPU2*0J

   E8 *+
f&R$
"__inference__wrapped_model_69685372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ä
ª
7__inference_batch_normalization_9_layer_call_fn_6976025

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69693342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
é
ú
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6970713

inputs
conv2d_13_6970697
conv2d_13_6970699"
batch_normalization_13_6970702"
batch_normalization_13_6970704"
batch_normalization_13_6970706"
batch_normalization_13_6970708
identity¢.batch_normalization_13/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall°
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_13_6970697conv2d_13_6970699*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_13_layer_call_and_return_conditional_losses_69705332#
!conv2d_13/StatefulPartitionedCallÙ
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_13_6970702batch_normalization_13_6970704batch_normalization_13_6970706batch_normalization_13_6970708*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_697058620
.batch_normalization_13/StatefulPartitionedCall
re_lu_13/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_13_layer_call_and_return_conditional_losses_69706272
re_lu_13/PartitionedCallÓ
IdentityIdentity!re_lu_13/PartitionedCall:output:0/^batch_normalization_13/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
« 

H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6974967

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource
identity¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1³
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_10/Conv2D/ReadVariableOpÂ
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_10/Conv2Dª
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp°
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_10/BiasAdd¹
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp¿
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1ì
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ö
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/BiasAdd:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_10/FusedBatchNormV3
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValue
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1
re_lu_10/ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_10/ReluÉ
IdentityIdentityre_lu_10/Relu:activations:0&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
« 

H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975311

inputs,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource
identity¢%batch_normalization_12/AssignNewValue¢'batch_normalization_12/AssignNewValue_1³
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_12/Conv2D/ReadVariableOpÂ
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_12/Conv2Dª
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp°
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_12/BiasAdd¹
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOp¿
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1ì
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ö
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_12/FusedBatchNormV3
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_12/AssignNewValue
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_12/AssignNewValue_1
re_lu_12/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_12/ReluÉ
IdentityIdentityre_lu_12/Relu:activations:0&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò
`
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6968749

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
«
8__inference_batch_normalization_10_layer_call_fn_6976169

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_69695382
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
«
Ç
-__inference_cnn_block_6_layer_call_fn_6975525
conv2d_13_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69706772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameconv2d_13_input
¨
®
F__inference_conv2d_11_layer_call_and_return_conditional_losses_6969907

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

X
<__inference_global_average_pooling2d_1_layer_call_fn_6970741

inputs
identityé
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *`
f[RY
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69707352
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6968599

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
®
F__inference_conv2d_12_layer_call_and_return_conditional_losses_6976423

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Å
ò
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6968835

inputs
conv2d_7_6968819
conv2d_7_6968821!
batch_normalization_7_6968824!
batch_normalization_7_6968826!
batch_normalization_7_6968828!
batch_normalization_7_6968830
identity¢-batch_normalization_7/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCallª
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_6968819conv2d_7_6968821*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_69686552"
 conv2d_7/StatefulPartitionedCallÐ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_6968824batch_normalization_7_6968826batch_normalization_7_6968828batch_normalization_7_6968830*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69687082/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_69687492
re_lu_7/PartitionedCallÏ
IdentityIdentity re_lu_7/PartitionedCall:output:0.^batch_normalization_7/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
°
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976673

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
­
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6968655

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

9__inference_feature_extractor_cnn_1_layer_call_fn_6973912

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**8
config_proto(&

CPU

GPU2*0J

   E8 *]
fXRV
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_69714562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
ª
7__inference_batch_normalization_9_layer_call_fn_6976012

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69693162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¾
-__inference_cnn_block_3_layer_call_fn_6975026

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69697742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Þ

S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6970586

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬'
æ
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6972023

inputs#
feature_extractor_cnn_1_6971934#
feature_extractor_cnn_1_6971936#
feature_extractor_cnn_1_6971938#
feature_extractor_cnn_1_6971940#
feature_extractor_cnn_1_6971942#
feature_extractor_cnn_1_6971944#
feature_extractor_cnn_1_6971946#
feature_extractor_cnn_1_6971948#
feature_extractor_cnn_1_6971950#
feature_extractor_cnn_1_6971952#
feature_extractor_cnn_1_6971954#
feature_extractor_cnn_1_6971956#
feature_extractor_cnn_1_6971958#
feature_extractor_cnn_1_6971960#
feature_extractor_cnn_1_6971962#
feature_extractor_cnn_1_6971964#
feature_extractor_cnn_1_6971966#
feature_extractor_cnn_1_6971968#
feature_extractor_cnn_1_6971970#
feature_extractor_cnn_1_6971972#
feature_extractor_cnn_1_6971974#
feature_extractor_cnn_1_6971976#
feature_extractor_cnn_1_6971978#
feature_extractor_cnn_1_6971980#
feature_extractor_cnn_1_6971982#
feature_extractor_cnn_1_6971984#
feature_extractor_cnn_1_6971986#
feature_extractor_cnn_1_6971988#
feature_extractor_cnn_1_6971990#
feature_extractor_cnn_1_6971992#
feature_extractor_cnn_1_6971994#
feature_extractor_cnn_1_6971996#
feature_extractor_cnn_1_6971998#
feature_extractor_cnn_1_6972000#
feature_extractor_cnn_1_6972002#
feature_extractor_cnn_1_6972004#
feature_extractor_cnn_1_6972006#
feature_extractor_cnn_1_6972008#
feature_extractor_cnn_1_6972010#
feature_extractor_cnn_1_6972012#
feature_extractor_cnn_1_6972014#
feature_extractor_cnn_1_6972016
dense_2_6972019
identity¢dense_2/StatefulPartitionedCall¢/feature_extractor_cnn_1/StatefulPartitionedCall[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/ys
truedivRealDivinputstruediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truedivÝ
/feature_extractor_cnn_1/StatefulPartitionedCallStatefulPartitionedCalltruediv:z:0feature_extractor_cnn_1_6971934feature_extractor_cnn_1_6971936feature_extractor_cnn_1_6971938feature_extractor_cnn_1_6971940feature_extractor_cnn_1_6971942feature_extractor_cnn_1_6971944feature_extractor_cnn_1_6971946feature_extractor_cnn_1_6971948feature_extractor_cnn_1_6971950feature_extractor_cnn_1_6971952feature_extractor_cnn_1_6971954feature_extractor_cnn_1_6971956feature_extractor_cnn_1_6971958feature_extractor_cnn_1_6971960feature_extractor_cnn_1_6971962feature_extractor_cnn_1_6971964feature_extractor_cnn_1_6971966feature_extractor_cnn_1_6971968feature_extractor_cnn_1_6971970feature_extractor_cnn_1_6971972feature_extractor_cnn_1_6971974feature_extractor_cnn_1_6971976feature_extractor_cnn_1_6971978feature_extractor_cnn_1_6971980feature_extractor_cnn_1_6971982feature_extractor_cnn_1_6971984feature_extractor_cnn_1_6971986feature_extractor_cnn_1_6971988feature_extractor_cnn_1_6971990feature_extractor_cnn_1_6971992feature_extractor_cnn_1_6971994feature_extractor_cnn_1_6971996feature_extractor_cnn_1_6971998feature_extractor_cnn_1_6972000feature_extractor_cnn_1_6972002feature_extractor_cnn_1_6972004feature_extractor_cnn_1_6972006feature_extractor_cnn_1_6972008feature_extractor_cnn_1_6972010feature_extractor_cnn_1_6972012feature_extractor_cnn_1_6972014feature_extractor_cnn_1_6972016*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
 !"%&'(*8
config_proto(&

CPU

GPU2*0J

   E8 *]
fXRV
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_697127121
/feature_extractor_cnn_1/StatefulPartitionedCall¼
dense_2/StatefulPartitionedCallStatefulPartitionedCall8feature_extractor_cnn_1/StatefulPartitionedCall:output:0dense_2_6972019*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_69718192!
dense_2/StatefulPartitionedCallÐ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall0^feature_extractor_cnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2b
/feature_extractor_cnn_1/StatefulPartitionedCall/feature_extractor_cnn_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
«
8__inference_batch_normalization_12_layer_call_fn_6976560

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_69702732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
Æ
-__inference_cnn_block_0_layer_call_fn_6974596
conv2d_7_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69688352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_7_input
ä
«
8__inference_batch_normalization_12_layer_call_fn_6976547

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_69702552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§

S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6970508

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
ª
7__inference_batch_normalization_7_layer_call_fn_6975762

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69686902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

9__inference_feature_extractor_cnn_1_layer_call_fn_6974321
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
 !"%&'(*8
config_proto(&

CPU

GPU2*0J

   E8 *]
fXRV
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_69712712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
·û
þ
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6973581

inputs7
3cnn_block_0_conv2d_7_conv2d_readvariableop_resource8
4cnn_block_0_conv2d_7_biasadd_readvariableop_resource=
9cnn_block_0_batch_normalization_7_readvariableop_resource?
;cnn_block_0_batch_normalization_7_readvariableop_1_resourceN
Jcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_1_conv2d_8_conv2d_readvariableop_resource8
4cnn_block_1_conv2d_8_biasadd_readvariableop_resource=
9cnn_block_1_batch_normalization_8_readvariableop_resource?
;cnn_block_1_batch_normalization_8_readvariableop_1_resourceN
Jcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_2_conv2d_9_conv2d_readvariableop_resource8
4cnn_block_2_conv2d_9_biasadd_readvariableop_resource=
9cnn_block_2_batch_normalization_9_readvariableop_resource?
;cnn_block_2_batch_normalization_9_readvariableop_1_resourceN
Jcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_3_conv2d_10_conv2d_readvariableop_resource9
5cnn_block_3_conv2d_10_biasadd_readvariableop_resource>
:cnn_block_3_batch_normalization_10_readvariableop_resource@
<cnn_block_3_batch_normalization_10_readvariableop_1_resourceO
Kcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_4_conv2d_11_conv2d_readvariableop_resource9
5cnn_block_4_conv2d_11_biasadd_readvariableop_resource>
:cnn_block_4_batch_normalization_11_readvariableop_resource@
<cnn_block_4_batch_normalization_11_readvariableop_1_resourceO
Kcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_5_conv2d_12_conv2d_readvariableop_resource9
5cnn_block_5_conv2d_12_biasadd_readvariableop_resource>
:cnn_block_5_batch_normalization_12_readvariableop_resource@
<cnn_block_5_batch_normalization_12_readvariableop_1_resourceO
Kcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_6_conv2d_13_conv2d_readvariableop_resource9
5cnn_block_6_conv2d_13_biasadd_readvariableop_resource>
:cnn_block_6_batch_normalization_13_readvariableop_resource@
<cnn_block_6_batch_normalization_13_readvariableop_1_resourceO
Kcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource
identity¢0cnn_block_0/batch_normalization_7/AssignNewValue¢2cnn_block_0/batch_normalization_7/AssignNewValue_1¢0cnn_block_1/batch_normalization_8/AssignNewValue¢2cnn_block_1/batch_normalization_8/AssignNewValue_1¢0cnn_block_2/batch_normalization_9/AssignNewValue¢2cnn_block_2/batch_normalization_9/AssignNewValue_1¢1cnn_block_3/batch_normalization_10/AssignNewValue¢3cnn_block_3/batch_normalization_10/AssignNewValue_1¢1cnn_block_4/batch_normalization_11/AssignNewValue¢3cnn_block_4/batch_normalization_11/AssignNewValue_1¢1cnn_block_5/batch_normalization_12/AssignNewValue¢3cnn_block_5/batch_normalization_12/AssignNewValue_1¢1cnn_block_6/batch_normalization_13/AssignNewValue¢3cnn_block_6/batch_normalization_13/AssignNewValue_1Ô
*cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*cnn_block_0/conv2d_7/Conv2D/ReadVariableOpã
cnn_block_0/conv2d_7/Conv2DConv2Dinputs2cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_0/conv2d_7/Conv2DË
+cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpÜ
cnn_block_0/conv2d_7/BiasAddBiasAdd$cnn_block_0/conv2d_7/Conv2D:output:03cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/conv2d_7/BiasAddÚ
0cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype022
0cnn_block_0/batch_normalization_7/ReadVariableOpà
2cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype024
2cnn_block_0/batch_normalization_7/ReadVariableOp_1
Acnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02C
Acnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
Ccnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02E
Ccnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_0/conv2d_7/BiasAdd:output:08cnn_block_0/batch_normalization_7/ReadVariableOp:value:0:cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_0/batch_normalization_7/FusedBatchNormV3Ë
0cnn_block_0/batch_normalization_7/AssignNewValueAssignVariableOpJcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource?cnn_block_0/batch_normalization_7/FusedBatchNormV3:batch_mean:0B^cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_0/batch_normalization_7/AssignNewValueÙ
2cnn_block_0/batch_normalization_7/AssignNewValue_1AssignVariableOpLcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_0/batch_normalization_7/FusedBatchNormV3:batch_variance:0D^cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_0/batch_normalization_7/AssignNewValue_1®
cnn_block_0/re_lu_7/ReluRelu6cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/re_lu_7/ReluÔ
*cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_1/conv2d_8/Conv2D/ReadVariableOp
cnn_block_1/conv2d_8/Conv2DConv2D&cnn_block_0/re_lu_7/Relu:activations:02cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_1/conv2d_8/Conv2DË
+cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpÜ
cnn_block_1/conv2d_8/BiasAddBiasAdd$cnn_block_1/conv2d_8/Conv2D:output:03cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/conv2d_8/BiasAddÚ
0cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_1/batch_normalization_8/ReadVariableOpà
2cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_1/batch_normalization_8/ReadVariableOp_1
Acnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
Ccnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_1/conv2d_8/BiasAdd:output:08cnn_block_1/batch_normalization_8/ReadVariableOp:value:0:cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_1/batch_normalization_8/FusedBatchNormV3Ë
0cnn_block_1/batch_normalization_8/AssignNewValueAssignVariableOpJcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource?cnn_block_1/batch_normalization_8/FusedBatchNormV3:batch_mean:0B^cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_1/batch_normalization_8/AssignNewValueÙ
2cnn_block_1/batch_normalization_8/AssignNewValue_1AssignVariableOpLcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_1/batch_normalization_8/FusedBatchNormV3:batch_variance:0D^cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_1/batch_normalization_8/AssignNewValue_1®
cnn_block_1/re_lu_8/ReluRelu6cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/re_lu_8/ReluÔ
*cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_2/conv2d_9/Conv2D/ReadVariableOp
cnn_block_2/conv2d_9/Conv2DConv2D&cnn_block_1/re_lu_8/Relu:activations:02cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_2/conv2d_9/Conv2DË
+cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpÜ
cnn_block_2/conv2d_9/BiasAddBiasAdd$cnn_block_2/conv2d_9/Conv2D:output:03cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/conv2d_9/BiasAddÚ
0cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOp9cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_2/batch_normalization_9/ReadVariableOpà
2cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp;cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_2/batch_normalization_9/ReadVariableOp_1
Acnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
Ccnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%cnn_block_2/conv2d_9/BiasAdd:output:08cnn_block_2/batch_normalization_9/ReadVariableOp:value:0:cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0Icnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_2/batch_normalization_9/FusedBatchNormV3Ë
0cnn_block_2/batch_normalization_9/AssignNewValueAssignVariableOpJcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource?cnn_block_2/batch_normalization_9/FusedBatchNormV3:batch_mean:0B^cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_2/batch_normalization_9/AssignNewValueÙ
2cnn_block_2/batch_normalization_9/AssignNewValue_1AssignVariableOpLcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_2/batch_normalization_9/FusedBatchNormV3:batch_variance:0D^cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_2/batch_normalization_9/AssignNewValue_1®
cnn_block_2/re_lu_9/ReluRelu6cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/re_lu_9/Relu×
+cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+cnn_block_3/conv2d_10/Conv2D/ReadVariableOp
cnn_block_3/conv2d_10/Conv2DConv2D&cnn_block_2/re_lu_9/Relu:activations:03cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_3/conv2d_10/Conv2DÎ
,cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpà
cnn_block_3/conv2d_10/BiasAddBiasAdd%cnn_block_3/conv2d_10/Conv2D:output:04cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/conv2d_10/BiasAddÝ
1cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOp:cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype023
1cnn_block_3/batch_normalization_10/ReadVariableOpã
3cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOp<cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype025
3cnn_block_3/batch_normalization_10/ReadVariableOp_1
Bcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
Dcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ê
3cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3&cnn_block_3/conv2d_10/BiasAdd:output:09cnn_block_3/batch_normalization_10/ReadVariableOp:value:0;cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0Jcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<25
3cnn_block_3/batch_normalization_10/FusedBatchNormV3Ñ
1cnn_block_3/batch_normalization_10/AssignNewValueAssignVariableOpKcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource@cnn_block_3/batch_normalization_10/FusedBatchNormV3:batch_mean:0C^cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1cnn_block_3/batch_normalization_10/AssignNewValueß
3cnn_block_3/batch_normalization_10/AssignNewValue_1AssignVariableOpMcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceDcnn_block_3/batch_normalization_10/FusedBatchNormV3:batch_variance:0E^cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3cnn_block_3/batch_normalization_10/AssignNewValue_1±
cnn_block_3/re_lu_10/ReluRelu7cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/re_lu_10/Relu×
+cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+cnn_block_4/conv2d_11/Conv2D/ReadVariableOp
cnn_block_4/conv2d_11/Conv2DConv2D'cnn_block_3/re_lu_10/Relu:activations:03cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_4/conv2d_11/Conv2DÎ
,cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpà
cnn_block_4/conv2d_11/BiasAddBiasAdd%cnn_block_4/conv2d_11/Conv2D:output:04cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/conv2d_11/BiasAddÝ
1cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOp:cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype023
1cnn_block_4/batch_normalization_11/ReadVariableOpã
3cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOp<cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3cnn_block_4/batch_normalization_11/ReadVariableOp_1
Bcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
Dcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ê
3cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&cnn_block_4/conv2d_11/BiasAdd:output:09cnn_block_4/batch_normalization_11/ReadVariableOp:value:0;cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0Jcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<25
3cnn_block_4/batch_normalization_11/FusedBatchNormV3Ñ
1cnn_block_4/batch_normalization_11/AssignNewValueAssignVariableOpKcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource@cnn_block_4/batch_normalization_11/FusedBatchNormV3:batch_mean:0C^cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1cnn_block_4/batch_normalization_11/AssignNewValueß
3cnn_block_4/batch_normalization_11/AssignNewValue_1AssignVariableOpMcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceDcnn_block_4/batch_normalization_11/FusedBatchNormV3:batch_variance:0E^cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3cnn_block_4/batch_normalization_11/AssignNewValue_1±
cnn_block_4/re_lu_11/ReluRelu7cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/re_lu_11/Relu×
+cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+cnn_block_5/conv2d_12/Conv2D/ReadVariableOp
cnn_block_5/conv2d_12/Conv2DConv2D'cnn_block_4/re_lu_11/Relu:activations:03cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_5/conv2d_12/Conv2DÎ
,cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpà
cnn_block_5/conv2d_12/BiasAddBiasAdd%cnn_block_5/conv2d_12/Conv2D:output:04cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/conv2d_12/BiasAddÝ
1cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOp:cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype023
1cnn_block_5/batch_normalization_12/ReadVariableOpã
3cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOp<cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3cnn_block_5/batch_normalization_12/ReadVariableOp_1
Bcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp
Dcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ê
3cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3&cnn_block_5/conv2d_12/BiasAdd:output:09cnn_block_5/batch_normalization_12/ReadVariableOp:value:0;cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0Jcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<25
3cnn_block_5/batch_normalization_12/FusedBatchNormV3Ñ
1cnn_block_5/batch_normalization_12/AssignNewValueAssignVariableOpKcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource@cnn_block_5/batch_normalization_12/FusedBatchNormV3:batch_mean:0C^cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1cnn_block_5/batch_normalization_12/AssignNewValueß
3cnn_block_5/batch_normalization_12/AssignNewValue_1AssignVariableOpMcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceDcnn_block_5/batch_normalization_12/FusedBatchNormV3:batch_variance:0E^cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3cnn_block_5/batch_normalization_12/AssignNewValue_1±
cnn_block_5/re_lu_12/ReluRelu7cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/re_lu_12/ReluØ
+cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+cnn_block_6/conv2d_13/Conv2D/ReadVariableOp
cnn_block_6/conv2d_13/Conv2DConv2D'cnn_block_5/re_lu_12/Relu:activations:03cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_6/conv2d_13/Conv2DÏ
,cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpá
cnn_block_6/conv2d_13/BiasAddBiasAdd%cnn_block_6/conv2d_13/Conv2D:output:04cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/conv2d_13/BiasAddÞ
1cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOp:cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype023
1cnn_block_6/batch_normalization_13/ReadVariableOpä
3cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOp<cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3cnn_block_6/batch_normalization_13/ReadVariableOp_1
Bcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp
Dcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Ï
3cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3&cnn_block_6/conv2d_13/BiasAdd:output:09cnn_block_6/batch_normalization_13/ReadVariableOp:value:0;cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0Jcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<25
3cnn_block_6/batch_normalization_13/FusedBatchNormV3Ñ
1cnn_block_6/batch_normalization_13/AssignNewValueAssignVariableOpKcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource@cnn_block_6/batch_normalization_13/FusedBatchNormV3:batch_mean:0C^cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1cnn_block_6/batch_normalization_13/AssignNewValueß
3cnn_block_6/batch_normalization_13/AssignNewValue_1AssignVariableOpMcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resourceDcnn_block_6/batch_normalization_13/FusedBatchNormV3:batch_variance:0E^cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3cnn_block_6/batch_normalization_13/AssignNewValue_1²
cnn_block_6/re_lu_13/ReluRelu7cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/re_lu_13/Relu·
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indicesâ
global_average_pooling2d_1/MeanMean'cnn_block_6/re_lu_13/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
global_average_pooling2d_1/MeanÝ
IdentityIdentity(global_average_pooling2d_1/Mean:output:01^cnn_block_0/batch_normalization_7/AssignNewValue3^cnn_block_0/batch_normalization_7/AssignNewValue_11^cnn_block_1/batch_normalization_8/AssignNewValue3^cnn_block_1/batch_normalization_8/AssignNewValue_11^cnn_block_2/batch_normalization_9/AssignNewValue3^cnn_block_2/batch_normalization_9/AssignNewValue_12^cnn_block_3/batch_normalization_10/AssignNewValue4^cnn_block_3/batch_normalization_10/AssignNewValue_12^cnn_block_4/batch_normalization_11/AssignNewValue4^cnn_block_4/batch_normalization_11/AssignNewValue_12^cnn_block_5/batch_normalization_12/AssignNewValue4^cnn_block_5/batch_normalization_12/AssignNewValue_12^cnn_block_6/batch_normalization_13/AssignNewValue4^cnn_block_6/batch_normalization_13/AssignNewValue_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::2d
0cnn_block_0/batch_normalization_7/AssignNewValue0cnn_block_0/batch_normalization_7/AssignNewValue2h
2cnn_block_0/batch_normalization_7/AssignNewValue_12cnn_block_0/batch_normalization_7/AssignNewValue_12d
0cnn_block_1/batch_normalization_8/AssignNewValue0cnn_block_1/batch_normalization_8/AssignNewValue2h
2cnn_block_1/batch_normalization_8/AssignNewValue_12cnn_block_1/batch_normalization_8/AssignNewValue_12d
0cnn_block_2/batch_normalization_9/AssignNewValue0cnn_block_2/batch_normalization_9/AssignNewValue2h
2cnn_block_2/batch_normalization_9/AssignNewValue_12cnn_block_2/batch_normalization_9/AssignNewValue_12f
1cnn_block_3/batch_normalization_10/AssignNewValue1cnn_block_3/batch_normalization_10/AssignNewValue2j
3cnn_block_3/batch_normalization_10/AssignNewValue_13cnn_block_3/batch_normalization_10/AssignNewValue_12f
1cnn_block_4/batch_normalization_11/AssignNewValue1cnn_block_4/batch_normalization_11/AssignNewValue2j
3cnn_block_4/batch_normalization_11/AssignNewValue_13cnn_block_4/batch_normalization_11/AssignNewValue_12f
1cnn_block_5/batch_normalization_12/AssignNewValue1cnn_block_5/batch_normalization_12/AssignNewValue2j
3cnn_block_5/batch_normalization_12/AssignNewValue_13cnn_block_5/batch_normalization_12/AssignNewValue_12f
1cnn_block_6/batch_normalization_13/AssignNewValue1cnn_block_6/batch_normalization_13/AssignNewValue2j
3cnn_block_6/batch_normalization_13/AssignNewValue_13cnn_block_6/batch_normalization_13/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
åì
²
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6973232

inputsO
Kfeature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceO
Kfeature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceO
Kfeature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/ys
truedivRealDivinputstruediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv
Bfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02D
Bfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp°
3feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2DConv2Dtruediv:z:0Jfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D
Cfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02J
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02[
Yfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02]
[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ý
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3ö
0feature_extractor_cnn_1/cnn_block_0/re_lu_7/ReluReluNfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu
Bfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02D
Bfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpã
3feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu:activations:0Jfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D
Cfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ý
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3ö
0feature_extractor_cnn_1/cnn_block_1/re_lu_8/ReluReluNfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu
Bfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02D
Bfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpã
3feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu:activations:0Jfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D
Cfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ý
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3ö
0feature_extractor_cnn_1/cnn_block_2/re_lu_9/ReluReluNfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu
Cfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpæ
4feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D
Dfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02F
Dfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 27
5feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02\
Zfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02^
\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ä
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3ù
1feature_extractor_cnn_1/cnn_block_3/re_lu_10/ReluReluOfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu
Cfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpç
4feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D
Dfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02^
\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ä
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3ù
1feature_extractor_cnn_1/cnn_block_4/re_lu_11/ReluReluOfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu
Cfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpç
4feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D
Dfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02^
\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ä
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3ù
1feature_extractor_cnn_1/cnn_block_5/re_lu_12/ReluReluOfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu 
Cfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpè
4feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D
Dfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpÁ
5feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd¦
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp¬
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1Ù
Zfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpß
\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02^
\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1é
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3ú
1feature_extractor_cnn_1/cnn_block_6/re_lu_13/ReluReluOfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1feature_extractor_cnn_1/cnn_block_6/re_lu_13/Reluç
Ifeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2K
Ifeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesÂ
7feature_extractor_cnn_1/global_average_pooling2d_1/MeanMean?feature_extractor_cnn_1/cnn_block_6/re_lu_13/Relu:activations:0Rfeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7feature_extractor_cnn_1/global_average_pooling2d_1/Mean¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_2/MatMul/ReadVariableOpÅ
dense_2/MatMulMatMul@feature_extractor_cnn_1/global_average_pooling2d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_2/MatMull
IdentityIdentitydense_2/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975888

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¾
-__inference_cnn_block_5_layer_call_fn_6975370

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69704002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6969882

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î±

#__inference__traced_restore_6977018
file_prefix3
/assignvariableop_contrastive_cnn_dense_2_kernel&
"assignvariableop_1_conv2d_7_kernel$
 assignvariableop_2_conv2d_7_bias2
.assignvariableop_3_batch_normalization_7_gamma1
-assignvariableop_4_batch_normalization_7_beta&
"assignvariableop_5_conv2d_8_kernel$
 assignvariableop_6_conv2d_8_bias2
.assignvariableop_7_batch_normalization_8_gamma1
-assignvariableop_8_batch_normalization_8_beta&
"assignvariableop_9_conv2d_9_kernel%
!assignvariableop_10_conv2d_9_bias3
/assignvariableop_11_batch_normalization_9_gamma2
.assignvariableop_12_batch_normalization_9_beta(
$assignvariableop_13_conv2d_10_kernel&
"assignvariableop_14_conv2d_10_bias4
0assignvariableop_15_batch_normalization_10_gamma3
/assignvariableop_16_batch_normalization_10_beta(
$assignvariableop_17_conv2d_11_kernel&
"assignvariableop_18_conv2d_11_bias4
0assignvariableop_19_batch_normalization_11_gamma3
/assignvariableop_20_batch_normalization_11_beta(
$assignvariableop_21_conv2d_12_kernel&
"assignvariableop_22_conv2d_12_bias4
0assignvariableop_23_batch_normalization_12_gamma3
/assignvariableop_24_batch_normalization_12_beta(
$assignvariableop_25_conv2d_13_kernel&
"assignvariableop_26_conv2d_13_bias4
0assignvariableop_27_batch_normalization_13_gamma3
/assignvariableop_28_batch_normalization_13_beta9
5assignvariableop_29_batch_normalization_7_moving_mean=
9assignvariableop_30_batch_normalization_7_moving_variance9
5assignvariableop_31_batch_normalization_8_moving_mean=
9assignvariableop_32_batch_normalization_8_moving_variance9
5assignvariableop_33_batch_normalization_9_moving_mean=
9assignvariableop_34_batch_normalization_9_moving_variance:
6assignvariableop_35_batch_normalization_10_moving_mean>
:assignvariableop_36_batch_normalization_10_moving_variance:
6assignvariableop_37_batch_normalization_11_moving_mean>
:assignvariableop_38_batch_normalization_11_moving_variance:
6assignvariableop_39_batch_normalization_12_moving_mean>
:assignvariableop_40_batch_normalization_12_moving_variance:
6assignvariableop_41_batch_normalization_13_moving_mean>
:assignvariableop_42_batch_normalization_13_moving_variance
identity_44¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ÿ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*
valueBþ,B&head/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesæ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity®
AssignVariableOpAssignVariableOp/assignvariableop_contrastive_cnn_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_7_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¥
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_7_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3³
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_7_gammaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4²
AssignVariableOp_4AssignVariableOp-assignvariableop_4_batch_normalization_7_betaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_8_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¥
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_8_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_8_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8²
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_8_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9§
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_9_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10©
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_9_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_9_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¶
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_9_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¬
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_10_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ª
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_10_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_10_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16·
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_10_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¬
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_11_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ª
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_11_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¸
AssignVariableOp_19AssignVariableOp0assignvariableop_19_batch_normalization_11_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20·
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_11_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¬
AssignVariableOp_21AssignVariableOp$assignvariableop_21_conv2d_12_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ª
AssignVariableOp_22AssignVariableOp"assignvariableop_22_conv2d_12_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¸
AssignVariableOp_23AssignVariableOp0assignvariableop_23_batch_normalization_12_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_12_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¬
AssignVariableOp_25AssignVariableOp$assignvariableop_25_conv2d_13_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ª
AssignVariableOp_26AssignVariableOp"assignvariableop_26_conv2d_13_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¸
AssignVariableOp_27AssignVariableOp0assignvariableop_27_batch_normalization_13_gammaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28·
AssignVariableOp_28AssignVariableOp/assignvariableop_28_batch_normalization_13_betaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29½
AssignVariableOp_29AssignVariableOp5assignvariableop_29_batch_normalization_7_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Á
AssignVariableOp_30AssignVariableOp9assignvariableop_30_batch_normalization_7_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31½
AssignVariableOp_31AssignVariableOp5assignvariableop_31_batch_normalization_8_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Á
AssignVariableOp_32AssignVariableOp9assignvariableop_32_batch_normalization_8_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33½
AssignVariableOp_33AssignVariableOp5assignvariableop_33_batch_normalization_9_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Á
AssignVariableOp_34AssignVariableOp9assignvariableop_34_batch_normalization_9_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¾
AssignVariableOp_35AssignVariableOp6assignvariableop_35_batch_normalization_10_moving_meanIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Â
AssignVariableOp_36AssignVariableOp:assignvariableop_36_batch_normalization_10_moving_varianceIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¾
AssignVariableOp_37AssignVariableOp6assignvariableop_37_batch_normalization_11_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Â
AssignVariableOp_38AssignVariableOp:assignvariableop_38_batch_normalization_11_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¾
AssignVariableOp_39AssignVariableOp6assignvariableop_39_batch_normalization_12_moving_meanIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Â
AssignVariableOp_40AssignVariableOp:assignvariableop_40_batch_normalization_12_moving_varianceIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¾
AssignVariableOp_41AssignVariableOp6assignvariableop_41_batch_normalization_13_moving_meanIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Â
AssignVariableOp_42AssignVariableOp:assignvariableop_42_batch_normalization_13_moving_varianceIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*Ã
_input_shapes±
®: :::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ß
¹$
"__inference__wrapped_model_6968537
input_1_
[contrastive_cnn_feature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resource`
\contrastive_cnn_feature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resourcee
acontrastive_cnn_feature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resourceg
ccontrastive_cnn_feature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resourcev
rcontrastive_cnn_feature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourcex
tcontrastive_cnn_feature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource_
[contrastive_cnn_feature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resource`
\contrastive_cnn_feature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resourcee
acontrastive_cnn_feature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resourceg
ccontrastive_cnn_feature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resourcev
rcontrastive_cnn_feature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourcex
tcontrastive_cnn_feature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource_
[contrastive_cnn_feature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resource`
\contrastive_cnn_feature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resourcee
acontrastive_cnn_feature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resourceg
ccontrastive_cnn_feature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resourcev
rcontrastive_cnn_feature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourcex
tcontrastive_cnn_feature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource`
\contrastive_cnn_feature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resourcea
]contrastive_cnn_feature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resourcef
bcontrastive_cnn_feature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resourceh
dcontrastive_cnn_feature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resourcew
scontrastive_cnn_feature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourcey
ucontrastive_cnn_feature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource`
\contrastive_cnn_feature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resourcea
]contrastive_cnn_feature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resourcef
bcontrastive_cnn_feature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resourceh
dcontrastive_cnn_feature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resourcew
scontrastive_cnn_feature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourcey
ucontrastive_cnn_feature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource`
\contrastive_cnn_feature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resourcea
]contrastive_cnn_feature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resourcef
bcontrastive_cnn_feature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resourceh
dcontrastive_cnn_feature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resourcew
scontrastive_cnn_feature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourcey
ucontrastive_cnn_feature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource`
\contrastive_cnn_feature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resourcea
]contrastive_cnn_feature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resourcef
bcontrastive_cnn_feature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resourceh
dcontrastive_cnn_feature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resourcew
scontrastive_cnn_feature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourcey
ucontrastive_cnn_feature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:
6contrastive_cnn_dense_2_matmul_readvariableop_resource
identity{
contrastive_cnn/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
contrastive_cnn/truediv/y¤
contrastive_cnn/truedivRealDivinput_1"contrastive_cnn/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
contrastive_cnn/truedivÌ
Rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOp[contrastive_cnn_feature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02T
Rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOpð
Ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2DConv2Dcontrastive_cnn/truediv:z:0Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2E
Ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2DÃ
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp\contrastive_cnn_feature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02U
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpü
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAddBiasAddLcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D:output:0[contrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAddÒ
Xcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOpacontrastive_cnn_feature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02Z
Xcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOpØ
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOpccontrastive_cnn_feature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02\
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1
icontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOprcontrastive_cnn_feature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02k
icontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
kcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOptcontrastive_cnn_feature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02m
kcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Í
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3Mcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd:output:0`contrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp:value:0bcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0qcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0scontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2\
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3¦
@contrastive_cnn/feature_extractor_cnn_1/cnn_block_0/re_lu_7/ReluRelu^contrastive_cnn/feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@contrastive_cnn/feature_extractor_cnn_1/cnn_block_0/re_lu_7/ReluÌ
Rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp[contrastive_cnn_feature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02T
Rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOp£
Ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2DConv2DNcontrastive_cnn/feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu:activations:0Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2E
Ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2DÃ
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp\contrastive_cnn_feature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02U
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpü
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAddBiasAddLcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D:output:0[contrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAddÒ
Xcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOpacontrastive_cnn_feature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02Z
Xcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOpØ
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOpccontrastive_cnn_feature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02\
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1
icontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOprcontrastive_cnn_feature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02k
icontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
kcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOptcontrastive_cnn_feature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02m
kcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Í
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3Mcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd:output:0`contrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp:value:0bcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0qcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0scontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2\
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3¦
@contrastive_cnn/feature_extractor_cnn_1/cnn_block_1/re_lu_8/ReluRelu^contrastive_cnn/feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2B
@contrastive_cnn/feature_extractor_cnn_1/cnn_block_1/re_lu_8/ReluÌ
Rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp[contrastive_cnn_feature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02T
Rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOp£
Ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2DConv2DNcontrastive_cnn/feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu:activations:0Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2E
Ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2DÃ
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp\contrastive_cnn_feature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02U
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpü
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAddBiasAddLcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D:output:0[contrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAddÒ
Xcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOpacontrastive_cnn_feature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02Z
Xcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOpØ
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOpccontrastive_cnn_feature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02\
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1
icontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOprcontrastive_cnn_feature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02k
icontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
kcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOptcontrastive_cnn_feature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02m
kcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Í
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3Mcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd:output:0`contrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp:value:0bcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0qcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0scontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2\
Zcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3¦
@contrastive_cnn/feature_extractor_cnn_1/cnn_block_2/re_lu_9/ReluRelu^contrastive_cnn/feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2B
@contrastive_cnn/feature_extractor_cnn_1/cnn_block_2/re_lu_9/ReluÏ
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp\contrastive_cnn_feature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02U
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOp¦
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2DConv2DNcontrastive_cnn/feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu:activations:0[contrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2F
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2DÆ
Tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp]contrastive_cnn_feature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02V
Tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp
Econtrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAddBiasAddMcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D:output:0\contrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2G
Econtrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAddÕ
Ycontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOpbcontrastive_cnn_feature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02[
Ycontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOpÛ
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOpdcontrastive_cnn_feature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1
jcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpscontrastive_cnn_feature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02l
jcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
lcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpucontrastive_cnn_feature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02n
lcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ô
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3Ncontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd:output:0acontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp:value:0ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2]
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3©
Acontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/re_lu_10/ReluRelu_contrastive_cnn/feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2C
Acontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/re_lu_10/ReluÏ
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp\contrastive_cnn_feature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02U
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOp§
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2DConv2DOcontrastive_cnn/feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu:activations:0[contrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2F
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2DÆ
Tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp]contrastive_cnn_feature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02V
Tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp
Econtrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAddBiasAddMcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D:output:0\contrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2G
Econtrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAddÕ
Ycontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOpbcontrastive_cnn_feature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02[
Ycontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOpÛ
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOpdcontrastive_cnn_feature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02]
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1
jcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpscontrastive_cnn_feature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02l
jcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
lcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpucontrastive_cnn_feature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02n
lcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ô
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3Ncontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd:output:0acontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp:value:0ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2]
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3©
Acontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/re_lu_11/ReluRelu_contrastive_cnn/feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2C
Acontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/re_lu_11/ReluÏ
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOp\contrastive_cnn_feature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02U
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOp§
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2DConv2DOcontrastive_cnn/feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu:activations:0[contrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2F
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2DÆ
Tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp]contrastive_cnn_feature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02V
Tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp
Econtrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAddBiasAddMcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D:output:0\contrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2G
Econtrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAddÕ
Ycontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOpbcontrastive_cnn_feature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02[
Ycontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOpÛ
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOpdcontrastive_cnn_feature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02]
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1
jcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpscontrastive_cnn_feature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02l
jcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp
lcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpucontrastive_cnn_feature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02n
lcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Ô
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3Ncontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd:output:0acontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp:value:0ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2]
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3©
Acontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/re_lu_12/ReluRelu_contrastive_cnn/feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2C
Acontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/re_lu_12/ReluÐ
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOp\contrastive_cnn_feature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02U
Scontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOp¨
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2DConv2DOcontrastive_cnn/feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu:activations:0[contrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2F
Dcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2DÇ
Tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp]contrastive_cnn_feature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02V
Tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp
Econtrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAddBiasAddMcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D:output:0\contrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
Econtrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAddÖ
Ycontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOpbcontrastive_cnn_feature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02[
Ycontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOpÜ
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOpdcontrastive_cnn_feature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02]
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1
jcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpscontrastive_cnn_feature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02l
jcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp
lcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpucontrastive_cnn_feature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02n
lcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Ù
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3Ncontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd:output:0acontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp:value:0ccontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0rcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0tcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2]
[contrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3ª
Acontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/re_lu_13/ReluRelu_contrastive_cnn/feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
Acontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/re_lu_13/Relu
Ycontrastive_cnn/feature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2[
Ycontrastive_cnn/feature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indices
Gcontrastive_cnn/feature_extractor_cnn_1/global_average_pooling2d_1/MeanMeanOcontrastive_cnn/feature_extractor_cnn_1/cnn_block_6/re_lu_13/Relu:activations:0bcontrastive_cnn/feature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
Gcontrastive_cnn/feature_extractor_cnn_1/global_average_pooling2d_1/MeanÖ
-contrastive_cnn/dense_2/MatMul/ReadVariableOpReadVariableOp6contrastive_cnn_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02/
-contrastive_cnn/dense_2/MatMul/ReadVariableOp
contrastive_cnn/dense_2/MatMulMatMulPcontrastive_cnn/feature_extractor_cnn_1/global_average_pooling2d_1/Mean:output:05contrastive_cnn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
contrastive_cnn/dense_2/MatMul|
IdentityIdentity(contrastive_cnn/dense_2/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ë
°
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6969538

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×
a
E__inference_re_lu_13_layer_call_and_return_conditional_losses_6976722

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
ò
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6968799

inputs
conv2d_7_6968783
conv2d_7_6968785!
batch_normalization_7_6968788!
batch_normalization_7_6968790!
batch_normalization_7_6968792!
batch_normalization_7_6968794
identity¢-batch_normalization_7/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCallª
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_6968783conv2d_7_6968785*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_69686552"
 conv2d_7/StatefulPartitionedCallÎ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_6968788batch_normalization_7_6968790batch_normalization_7_6968792batch_normalization_7_6968794*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69686902/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_7_layer_call_and_return_conditional_losses_69687492
re_lu_7/PartitionedCallÏ
IdentityIdentity re_lu_7/PartitionedCall:output:0.^batch_normalization_7/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
ª
7__inference_batch_normalization_9_layer_call_fn_6976089

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69692562
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§
­
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6968968

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6976045

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

´
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974648

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp¿
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_8/BiasAdd¶
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp¼
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1é
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3
re_lu_8/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_8/Reluv
IdentityIdentityre_lu_8/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6969960

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6970195

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½
º
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975594

inputs,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource
identity´
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_13/Conv2D/ReadVariableOpÃ
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_13/Conv2D«
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp±
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_13/BiasAddº
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_13/ReadVariableOpÀ
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_13/ReadVariableOp_1í
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1í
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_13/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3
re_lu_13/ReluRelu+batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_13/Relux
IdentityIdentityre_lu_13/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@:::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò
`
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6975780

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6969316

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¾
-__inference_cnn_block_2_layer_call_fn_6974923

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69694252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
º
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6974992

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource
identity³
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_10/Conv2D/ReadVariableOpÂ
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_10/Conv2Dª
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp°
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_10/BiasAdd¹
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp¿
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1ì
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1è
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/BiasAdd:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3
re_lu_10/ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_10/Reluw
IdentityIdentityre_lu_10/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Í

9__inference_feature_extractor_cnn_1_layer_call_fn_6973823

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
 !"%&'(*8
config_proto(&

CPU

GPU2*0J

   E8 *]
fXRV
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_69712712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
®
F__inference_conv2d_13_layer_call_and_return_conditional_losses_6976580

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¾
-__inference_cnn_block_2_layer_call_fn_6974940

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69694612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
É
Ã
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975250
conv2d_11_input,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identity³
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_11/Conv2D/ReadVariableOpË
conv2d_11/Conv2DConv2Dconv2d_11_input'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_11/Conv2Dª
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp°
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_11/BiasAdd¹
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOp¿
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1ì
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1è
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3
re_lu_11/ReluRelu+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_11/Reluw
IdentityIdentityre_lu_11/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :::::::` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)
_user_specified_nameconv2d_11_input

¾
-__inference_cnn_block_4_layer_call_fn_6975181

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69700512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

1__inference_contrastive_cnn_layer_call_fn_6973414

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*8
config_proto(&

CPU

GPU2*0J

   E8 *U
fPRN
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_69722082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
º
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975336

inputs,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource2
.batch_normalization_12_readvariableop_resource4
0batch_normalization_12_readvariableop_1_resourceC
?batch_normalization_12_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource
identity³
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_12/Conv2D/ReadVariableOpÂ
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_12/Conv2Dª
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp°
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_12/BiasAdd¹
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_12/ReadVariableOp¿
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_12/ReadVariableOp_1ì
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1è
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_12/BiasAdd:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_12/FusedBatchNormV3
re_lu_12/ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_12/Reluw
IdentityIdentityre_lu_12/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@:::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ø
Ã
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975508
conv2d_13_input,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource
identity´
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_13/Conv2D/ReadVariableOpÌ
conv2d_13/Conv2DConv2Dconv2d_13_input'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_13/Conv2D«
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp±
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_13/BiasAddº
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_13/ReadVariableOpÀ
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_13/ReadVariableOp_1í
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1í
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_13/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_13/FusedBatchNormV3
re_lu_13/ReluRelu+batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_13/Relux
IdentityIdentityre_lu_13/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@:::::::` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameconv2d_13_input


S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976470

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


*__inference_conv2d_7_layer_call_fn_6975647

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_69686552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
ª
7__inference_batch_normalization_7_layer_call_fn_6975711

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69686302
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
o
)__inference_dense_2_layer_call_fn_6974424

inputs
unknown
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*#
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_69718192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
Ã
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6975078
conv2d_10_input,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource2
.batch_normalization_10_readvariableop_resource4
0batch_normalization_10_readvariableop_1_resourceC
?batch_normalization_10_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource
identity³
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_10/Conv2D/ReadVariableOpË
conv2d_10/Conv2DConv2Dconv2d_10_input'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_10/Conv2Dª
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp°
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_10/BiasAdd¹
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_10/ReadVariableOp¿
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_10/ReadVariableOp_1ì
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1è
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/BiasAdd:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3
re_lu_10/ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_10/Reluw
IdentityIdentityre_lu_10/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :::::::` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)
_user_specified_nameconv2d_10_input
Ò
`
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6969062

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
©
Ç
-__inference_cnn_block_3_layer_call_fn_6975095
conv2d_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69697382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)
_user_specified_nameconv2d_10_input


R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6969256

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°
«
8__inference_batch_normalization_13_layer_call_fn_6976704

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_69704772
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_9_layer_call_fn_6975961

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_69692812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ò
`
D__inference_re_lu_9_layer_call_and_return_conditional_losses_6969375

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
è
«
8__inference_batch_normalization_13_layer_call_fn_6976640

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_69705682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6970255

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ

9__inference_feature_extractor_cnn_1_layer_call_fn_6974410
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**8
config_proto(&

CPU

GPU2*0J

   E8 *]
fXRV
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_69714562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
©
Ç
-__inference_cnn_block_5_layer_call_fn_6975439
conv2d_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69703642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameconv2d_12_input
º 

H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975569

inputs,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource2
.batch_normalization_13_readvariableop_resource4
0batch_normalization_13_readvariableop_1_resourceC
?batch_normalization_13_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource
identity¢%batch_normalization_13/AssignNewValue¢'batch_normalization_13/AssignNewValue_1´
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_13/Conv2D/ReadVariableOpÃ
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_13/Conv2D«
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp±
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_13/BiasAddº
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_13/ReadVariableOpÀ
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_13/ReadVariableOp_1í
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1û
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_13/BiasAdd:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_13/FusedBatchNormV3
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_13/AssignNewValue
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_13/AssignNewValue_1
re_lu_13/ReluRelu+batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_13/ReluÊ
IdentityIdentityre_lu_13/Relu:activations:0&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
­
Ç
-__inference_cnn_block_6_layer_call_fn_6975542
conv2d_13_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallconv2d_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69707132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
)
_user_specified_nameconv2d_13_input


R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975842

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿
F
*__inference_re_lu_10_layer_call_fn_6976256

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_69696882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬
«
8__inference_batch_normalization_11_layer_call_fn_6976326

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_69698512
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975749

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974709
conv2d_8_input+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOpÇ
conv2d_8/Conv2DConv2Dconv2d_8_input&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_8/BiasAdd¶
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp¼
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1é
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_8/FusedBatchNormV3
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1
re_lu_8/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_8/ReluÆ
IdentityIdentityre_lu_8/Relu:activations:0%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_8_input

´
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974476

inputs+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOp¿
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_7/Conv2D§
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp¬
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_7/BiasAdd¶
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp¼
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1é
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
re_lu_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_7/Reluv
IdentityIdentityre_lu_7/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÌÒ
â'
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6973074

inputsO
Kfeature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceO
Kfeature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceO
Kfeature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resourceP
Lfeature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resourceU
Qfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resourceW
Sfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resourcef
bfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceh
dfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceP
Lfeature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resourceQ
Mfeature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resourceV
Rfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resourceX
Tfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resourceg
cfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourcei
efeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource
identity¢Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue¢Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1¢Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue¢Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1¢Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue¢Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1¢Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue¢Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1¢Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue¢Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1¢Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue¢Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1¢Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue¢Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/ys
truedivRealDivinputstruediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv
Bfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02D
Bfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp°
3feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2DConv2Dtruediv:z:0Jfeature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D
Cfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_0/conv2d_7/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02J
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02[
Yfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02]
[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ë
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_0/conv2d_7/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3Û
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValueAssignVariableOpbfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceWfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:batch_mean:0Z^feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*u
_classk
igloc:@feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValueé
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1AssignVariableOpdfeature_extractor_cnn_1_cnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource[feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:batch_variance:0\^feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*w
_classm
kiloc:@feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1ö
0feature_extractor_cnn_1/cnn_block_0/re_lu_7/ReluReluNfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu
Bfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02D
Bfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOpã
3feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_0/re_lu_7/Relu:activations:0Jfeature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D
Cfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_1/conv2d_8/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ë
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_1/conv2d_8/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3Û
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValueAssignVariableOpbfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceWfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:batch_mean:0Z^feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*u
_classk
igloc:@feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValueé
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1AssignVariableOpdfeature_extractor_cnn_1_cnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource[feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:batch_variance:0\^feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*w
_classm
kiloc:@feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1ö
0feature_extractor_cnn_1/cnn_block_1/re_lu_8/ReluReluNfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu
Bfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOpKfeature_extractor_cnn_1_cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02D
Bfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOpã
3feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_1/re_lu_8/Relu:activations:0Jfeature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
25
3feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D
Cfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp¼
4feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAddBiasAdd<feature_extractor_cnn_1/cnn_block_2/conv2d_9/Conv2D:output:0Kfeature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 26
4feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd¢
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOpQfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp¨
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOpSfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1Õ
Yfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpbfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpÛ
[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpdfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ë
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3=feature_extractor_cnn_1/cnn_block_2/conv2d_9/BiasAdd:output:0Pfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp:value:0Rfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0afeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0cfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3Û
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValueAssignVariableOpbfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceWfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:batch_mean:0Z^feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*u
_classk
igloc:@feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValueé
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1AssignVariableOpdfeature_extractor_cnn_1_cnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource[feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:batch_variance:0\^feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*w
_classm
kiloc:@feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02L
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1ö
0feature_extractor_cnn_1/cnn_block_2/re_lu_9/ReluReluNfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu
Cfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02E
Cfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOpæ
4feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2DConv2D>feature_extractor_cnn_1/cnn_block_2/re_lu_9/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D
Dfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02F
Dfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_3/conv2d_10/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 27
5feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02\
Zfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02^
\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ò
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_3/conv2d_10/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3á
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValueAssignVariableOpcfeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceXfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:batch_mean:0[^feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp*v
_classl
jhloc:@feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValueï
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1AssignVariableOpefeature_extractor_cnn_1_cnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource\feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:batch_variance:0]^feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*x
_classn
ljloc:@feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1ù
1feature_extractor_cnn_1/cnn_block_3/re_lu_10/ReluReluOfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu
Cfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOpç
4feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_3/re_lu_10/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D
Dfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_4/conv2d_11/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02^
\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ò
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_4/conv2d_11/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3á
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValueAssignVariableOpcfeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceXfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:batch_mean:0[^feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp*v
_classl
jhloc:@feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValueï
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1AssignVariableOpefeature_extractor_cnn_1_cnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource\feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:batch_variance:0]^feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*x
_classn
ljloc:@feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1ù
1feature_extractor_cnn_1/cnn_block_4/re_lu_11/ReluReluOfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu
Cfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOpç
4feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_4/re_lu_11/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D
Dfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpÀ
5feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_5/conv2d_12/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@27
5feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd¥
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp«
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1Ø
Zfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpÞ
\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02^
\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ò
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_5/conv2d_12/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3á
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValueAssignVariableOpcfeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceXfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:batch_mean:0[^feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp*v
_classl
jhloc:@feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValueï
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1AssignVariableOpefeature_extractor_cnn_1_cnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource\feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:batch_variance:0]^feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*x
_classn
ljloc:@feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1ù
1feature_extractor_cnn_1/cnn_block_5/re_lu_12/ReluReluOfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@23
1feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu 
Cfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOpLfeature_extractor_cnn_1_cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02E
Cfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOpè
4feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2DConv2D?feature_extractor_cnn_1/cnn_block_5/re_lu_12/Relu:activations:0Kfeature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
26
4feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D
Dfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpMfeature_extractor_cnn_1_cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02F
Dfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpÁ
5feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAddBiasAdd=feature_extractor_cnn_1/cnn_block_6/conv2d_13/Conv2D:output:0Lfeature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd¦
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOpRfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp¬
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOpTfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype02M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1Ù
Zfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpcfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02\
Zfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpß
\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpefeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02^
\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1÷
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3>feature_extractor_cnn_1/cnn_block_6/conv2d_13/BiasAdd:output:0Qfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp:value:0Sfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0bfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0dfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3á
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValueAssignVariableOpcfeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceXfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:batch_mean:0[^feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp*v
_classl
jhloc:@feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02K
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValueï
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1AssignVariableOpefeature_extractor_cnn_1_cnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource\feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:batch_variance:0]^feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*x
_classn
ljloc:@feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02M
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1ú
1feature_extractor_cnn_1/cnn_block_6/re_lu_13/ReluReluOfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1feature_extractor_cnn_1/cnn_block_6/re_lu_13/Reluç
Ifeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2K
Ifeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indicesÂ
7feature_extractor_cnn_1/global_average_pooling2d_1/MeanMean?feature_extractor_cnn_1/cnn_block_6/re_lu_13/Relu:activations:0Rfeature_extractor_cnn_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7feature_extractor_cnn_1/global_average_pooling2d_1/Mean¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_2/MatMul/ReadVariableOpÅ
dense_2/MatMulMatMul@feature_extractor_cnn_1/global_average_pooling2d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_2/MatMul	
IdentityIdentitydense_2/MatMul:product:0I^feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValueK^feature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1I^feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValueK^feature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1I^feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValueK^feature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1J^feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValueL^feature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1J^feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValueL^feature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1J^feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValueL^feature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1J^feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValueL^feature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::2
Hfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValueHfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue2
Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_1Jfeature_extractor_cnn_1/cnn_block_0/batch_normalization_7/AssignNewValue_12
Hfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValueHfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue2
Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_1Jfeature_extractor_cnn_1/cnn_block_1/batch_normalization_8/AssignNewValue_12
Hfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValueHfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue2
Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_1Jfeature_extractor_cnn_1/cnn_block_2/batch_normalization_9/AssignNewValue_12
Ifeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValueIfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue2
Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_1Kfeature_extractor_cnn_1/cnn_block_3/batch_normalization_10/AssignNewValue_12
Ifeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValueIfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue2
Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_1Kfeature_extractor_cnn_1/cnn_block_4/batch_normalization_11/AssignNewValue_12
Ifeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValueIfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue2
Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_1Kfeature_extractor_cnn_1/cnn_block_5/batch_normalization_12/AssignNewValue_12
Ifeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValueIfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue2
Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1Kfeature_extractor_cnn_1/cnn_block_6/batch_normalization_13/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

´
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974906

inputs+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_9/Conv2D/ReadVariableOp¿
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_9/Conv2D§
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp¬
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_9/BiasAdd¶
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp¼
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1é
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_9/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3
re_lu_9/ReluRelu*batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_9/Reluv
IdentityIdentityre_lu_9/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ò

S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976220

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
a
E__inference_re_lu_11_layer_call_and_return_conditional_losses_6970001

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦
Æ
-__inference_cnn_block_1_layer_call_fn_6974751
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69691122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_8_input
å
ú
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6969774

inputs
conv2d_10_6969758
conv2d_10_6969760"
batch_normalization_10_6969763"
batch_normalization_10_6969765"
batch_normalization_10_6969767"
batch_normalization_10_6969769
identity¢.batch_normalization_10/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¯
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_6969758conv2d_10_6969760*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_69695942#
!conv2d_10/StatefulPartitionedCallØ
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_6969763batch_normalization_10_6969765batch_normalization_10_6969767batch_normalization_10_6969769*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_696964720
.batch_normalization_10/StatefulPartitionedCall
re_lu_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_10_layer_call_and_return_conditional_losses_69696882
re_lu_10/PartitionedCallÒ
IdentityIdentity!re_lu_10/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨
®
F__inference_conv2d_11_layer_call_and_return_conditional_losses_6976266

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
a
E__inference_re_lu_12_layer_call_and_return_conditional_losses_6970314

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6975999

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
þ¯

T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6973734

inputs7
3cnn_block_0_conv2d_7_conv2d_readvariableop_resource8
4cnn_block_0_conv2d_7_biasadd_readvariableop_resource=
9cnn_block_0_batch_normalization_7_readvariableop_resource?
;cnn_block_0_batch_normalization_7_readvariableop_1_resourceN
Jcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_1_conv2d_8_conv2d_readvariableop_resource8
4cnn_block_1_conv2d_8_biasadd_readvariableop_resource=
9cnn_block_1_batch_normalization_8_readvariableop_resource?
;cnn_block_1_batch_normalization_8_readvariableop_1_resourceN
Jcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_2_conv2d_9_conv2d_readvariableop_resource8
4cnn_block_2_conv2d_9_biasadd_readvariableop_resource=
9cnn_block_2_batch_normalization_9_readvariableop_resource?
;cnn_block_2_batch_normalization_9_readvariableop_1_resourceN
Jcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_3_conv2d_10_conv2d_readvariableop_resource9
5cnn_block_3_conv2d_10_biasadd_readvariableop_resource>
:cnn_block_3_batch_normalization_10_readvariableop_resource@
<cnn_block_3_batch_normalization_10_readvariableop_1_resourceO
Kcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_4_conv2d_11_conv2d_readvariableop_resource9
5cnn_block_4_conv2d_11_biasadd_readvariableop_resource>
:cnn_block_4_batch_normalization_11_readvariableop_resource@
<cnn_block_4_batch_normalization_11_readvariableop_1_resourceO
Kcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_5_conv2d_12_conv2d_readvariableop_resource9
5cnn_block_5_conv2d_12_biasadd_readvariableop_resource>
:cnn_block_5_batch_normalization_12_readvariableop_resource@
<cnn_block_5_batch_normalization_12_readvariableop_1_resourceO
Kcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8
4cnn_block_6_conv2d_13_conv2d_readvariableop_resource9
5cnn_block_6_conv2d_13_biasadd_readvariableop_resource>
:cnn_block_6_batch_normalization_13_readvariableop_resource@
<cnn_block_6_batch_normalization_13_readvariableop_1_resourceO
Kcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceQ
Mcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource
identityÔ
*cnn_block_0/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3cnn_block_0_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*cnn_block_0/conv2d_7/Conv2D/ReadVariableOpã
cnn_block_0/conv2d_7/Conv2DConv2Dinputs2cnn_block_0/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_0/conv2d_7/Conv2DË
+cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_0_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+cnn_block_0/conv2d_7/BiasAdd/ReadVariableOpÜ
cnn_block_0/conv2d_7/BiasAddBiasAdd$cnn_block_0/conv2d_7/Conv2D:output:03cnn_block_0/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/conv2d_7/BiasAddÚ
0cnn_block_0/batch_normalization_7/ReadVariableOpReadVariableOp9cnn_block_0_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype022
0cnn_block_0/batch_normalization_7/ReadVariableOpà
2cnn_block_0/batch_normalization_7/ReadVariableOp_1ReadVariableOp;cnn_block_0_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype024
2cnn_block_0/batch_normalization_7/ReadVariableOp_1
Acnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02C
Acnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
Ccnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_0_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02E
Ccnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_0/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%cnn_block_0/conv2d_7/BiasAdd:output:08cnn_block_0/batch_normalization_7/ReadVariableOp:value:0:cnn_block_0/batch_normalization_7/ReadVariableOp_1:value:0Icnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_0/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 24
2cnn_block_0/batch_normalization_7/FusedBatchNormV3®
cnn_block_0/re_lu_7/ReluRelu6cnn_block_0/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/re_lu_7/ReluÔ
*cnn_block_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3cnn_block_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_1/conv2d_8/Conv2D/ReadVariableOp
cnn_block_1/conv2d_8/Conv2DConv2D&cnn_block_0/re_lu_7/Relu:activations:02cnn_block_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_1/conv2d_8/Conv2DË
+cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_1/conv2d_8/BiasAdd/ReadVariableOpÜ
cnn_block_1/conv2d_8/BiasAddBiasAdd$cnn_block_1/conv2d_8/Conv2D:output:03cnn_block_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/conv2d_8/BiasAddÚ
0cnn_block_1/batch_normalization_8/ReadVariableOpReadVariableOp9cnn_block_1_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_1/batch_normalization_8/ReadVariableOpà
2cnn_block_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp;cnn_block_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_1/batch_normalization_8/ReadVariableOp_1
Acnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
Ccnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%cnn_block_1/conv2d_8/BiasAdd:output:08cnn_block_1/batch_normalization_8/ReadVariableOp:value:0:cnn_block_1/batch_normalization_8/ReadVariableOp_1:value:0Icnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_1/batch_normalization_8/FusedBatchNormV3®
cnn_block_1/re_lu_8/ReluRelu6cnn_block_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/re_lu_8/ReluÔ
*cnn_block_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp3cnn_block_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_2/conv2d_9/Conv2D/ReadVariableOp
cnn_block_2/conv2d_9/Conv2DConv2D&cnn_block_1/re_lu_8/Relu:activations:02cnn_block_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_2/conv2d_9/Conv2DË
+cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_2/conv2d_9/BiasAdd/ReadVariableOpÜ
cnn_block_2/conv2d_9/BiasAddBiasAdd$cnn_block_2/conv2d_9/Conv2D:output:03cnn_block_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/conv2d_9/BiasAddÚ
0cnn_block_2/batch_normalization_9/ReadVariableOpReadVariableOp9cnn_block_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_2/batch_normalization_9/ReadVariableOpà
2cnn_block_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp;cnn_block_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_2/batch_normalization_9/ReadVariableOp_1
Acnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp
Ccnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%cnn_block_2/conv2d_9/BiasAdd:output:08cnn_block_2/batch_normalization_9/ReadVariableOp:value:0:cnn_block_2/batch_normalization_9/ReadVariableOp_1:value:0Icnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_2/batch_normalization_9/FusedBatchNormV3®
cnn_block_2/re_lu_9/ReluRelu6cnn_block_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/re_lu_9/Relu×
+cnn_block_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+cnn_block_3/conv2d_10/Conv2D/ReadVariableOp
cnn_block_3/conv2d_10/Conv2DConv2D&cnn_block_2/re_lu_9/Relu:activations:03cnn_block_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_3/conv2d_10/Conv2DÎ
,cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,cnn_block_3/conv2d_10/BiasAdd/ReadVariableOpà
cnn_block_3/conv2d_10/BiasAddBiasAdd%cnn_block_3/conv2d_10/Conv2D:output:04cnn_block_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/conv2d_10/BiasAddÝ
1cnn_block_3/batch_normalization_10/ReadVariableOpReadVariableOp:cnn_block_3_batch_normalization_10_readvariableop_resource*
_output_shapes
: *
dtype023
1cnn_block_3/batch_normalization_10/ReadVariableOpã
3cnn_block_3/batch_normalization_10/ReadVariableOp_1ReadVariableOp<cnn_block_3_batch_normalization_10_readvariableop_1_resource*
_output_shapes
: *
dtype025
3cnn_block_3/batch_normalization_10/ReadVariableOp_1
Bcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp
Dcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_3_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1¼
3cnn_block_3/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3&cnn_block_3/conv2d_10/BiasAdd:output:09cnn_block_3/batch_normalization_10/ReadVariableOp:value:0;cnn_block_3/batch_normalization_10/ReadVariableOp_1:value:0Jcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_3/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 25
3cnn_block_3/batch_normalization_10/FusedBatchNormV3±
cnn_block_3/re_lu_10/ReluRelu7cnn_block_3/batch_normalization_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/re_lu_10/Relu×
+cnn_block_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+cnn_block_4/conv2d_11/Conv2D/ReadVariableOp
cnn_block_4/conv2d_11/Conv2DConv2D'cnn_block_3/re_lu_10/Relu:activations:03cnn_block_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_4/conv2d_11/Conv2DÎ
,cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,cnn_block_4/conv2d_11/BiasAdd/ReadVariableOpà
cnn_block_4/conv2d_11/BiasAddBiasAdd%cnn_block_4/conv2d_11/Conv2D:output:04cnn_block_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/conv2d_11/BiasAddÝ
1cnn_block_4/batch_normalization_11/ReadVariableOpReadVariableOp:cnn_block_4_batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype023
1cnn_block_4/batch_normalization_11/ReadVariableOpã
3cnn_block_4/batch_normalization_11/ReadVariableOp_1ReadVariableOp<cnn_block_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3cnn_block_4/batch_normalization_11/ReadVariableOp_1
Bcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp
Dcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1¼
3cnn_block_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&cnn_block_4/conv2d_11/BiasAdd:output:09cnn_block_4/batch_normalization_11/ReadVariableOp:value:0;cnn_block_4/batch_normalization_11/ReadVariableOp_1:value:0Jcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 25
3cnn_block_4/batch_normalization_11/FusedBatchNormV3±
cnn_block_4/re_lu_11/ReluRelu7cnn_block_4/batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/re_lu_11/Relu×
+cnn_block_5/conv2d_12/Conv2D/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+cnn_block_5/conv2d_12/Conv2D/ReadVariableOp
cnn_block_5/conv2d_12/Conv2DConv2D'cnn_block_4/re_lu_11/Relu:activations:03cnn_block_5/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_5/conv2d_12/Conv2DÎ
,cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_5_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,cnn_block_5/conv2d_12/BiasAdd/ReadVariableOpà
cnn_block_5/conv2d_12/BiasAddBiasAdd%cnn_block_5/conv2d_12/Conv2D:output:04cnn_block_5/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/conv2d_12/BiasAddÝ
1cnn_block_5/batch_normalization_12/ReadVariableOpReadVariableOp:cnn_block_5_batch_normalization_12_readvariableop_resource*
_output_shapes
:@*
dtype023
1cnn_block_5/batch_normalization_12/ReadVariableOpã
3cnn_block_5/batch_normalization_12/ReadVariableOp_1ReadVariableOp<cnn_block_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3cnn_block_5/batch_normalization_12/ReadVariableOp_1
Bcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp
Dcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1¼
3cnn_block_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3&cnn_block_5/conv2d_12/BiasAdd:output:09cnn_block_5/batch_normalization_12/ReadVariableOp:value:0;cnn_block_5/batch_normalization_12/ReadVariableOp_1:value:0Jcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 25
3cnn_block_5/batch_normalization_12/FusedBatchNormV3±
cnn_block_5/re_lu_12/ReluRelu7cnn_block_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/re_lu_12/ReluØ
+cnn_block_6/conv2d_13/Conv2D/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+cnn_block_6/conv2d_13/Conv2D/ReadVariableOp
cnn_block_6/conv2d_13/Conv2DConv2D'cnn_block_5/re_lu_12/Relu:activations:03cnn_block_6/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_6/conv2d_13/Conv2DÏ
,cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp5cnn_block_6_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,cnn_block_6/conv2d_13/BiasAdd/ReadVariableOpá
cnn_block_6/conv2d_13/BiasAddBiasAdd%cnn_block_6/conv2d_13/Conv2D:output:04cnn_block_6/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/conv2d_13/BiasAddÞ
1cnn_block_6/batch_normalization_13/ReadVariableOpReadVariableOp:cnn_block_6_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype023
1cnn_block_6/batch_normalization_13/ReadVariableOpä
3cnn_block_6/batch_normalization_13/ReadVariableOp_1ReadVariableOp<cnn_block_6_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3cnn_block_6/batch_normalization_13/ReadVariableOp_1
Bcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp
Dcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_block_6_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Á
3cnn_block_6/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3&cnn_block_6/conv2d_13/BiasAdd:output:09cnn_block_6/batch_normalization_13/ReadVariableOp:value:0;cnn_block_6/batch_normalization_13/ReadVariableOp_1:value:0Jcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Lcnn_block_6/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 25
3cnn_block_6/batch_normalization_13/FusedBatchNormV3²
cnn_block_6/re_lu_13/ReluRelu7cnn_block_6/batch_normalization_13/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/re_lu_13/Relu·
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indicesâ
global_average_pooling2d_1/MeanMean'cnn_block_6/re_lu_13/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
global_average_pooling2d_1/Mean}
IdentityIdentity(global_average_pooling2d_1/Mean:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6969003

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ä
ª
7__inference_batch_normalization_7_layer_call_fn_6975775

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69687082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

1__inference_contrastive_cnn_layer_call_fn_6973323

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*?
_read_only_resource_inputs!
	
 !"%&'(+*8
config_proto(&

CPU

GPU2*0J

   E8 *U
fPRN
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_69720232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*Ü
_input_shapesÊ
Ç:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
º
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975164

inputs,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identity³
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_11/Conv2D/ReadVariableOpÂ
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_11/Conv2Dª
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp°
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_11/BiasAdd¹
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_11/ReadVariableOp¿
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_11/ReadVariableOp_1ì
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1è
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3
re_lu_11/ReluRelu+batch_normalization_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_11/Reluw
IdentityIdentityre_lu_11/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¾
-__inference_cnn_block_6_layer_call_fn_6975628

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69707132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
®
F__inference_conv2d_10_layer_call_and_return_conditional_losses_6969594

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6969569

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6969021

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ã
ò
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6969425

inputs
conv2d_9_6969409
conv2d_9_6969411!
batch_normalization_9_6969414!
batch_normalization_9_6969416!
batch_normalization_9_6969418!
batch_normalization_9_6969420
identity¢-batch_normalization_9/StatefulPartitionedCall¢ conv2d_9/StatefulPartitionedCallª
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_6969409conv2d_9_6969411*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_69692812"
 conv2d_9/StatefulPartitionedCallÎ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_9_6969414batch_normalization_9_6969416batch_normalization_9_6969418batch_normalization_9_6969420*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69693162/
-batch_normalization_9/StatefulPartitionedCall
re_lu_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_re_lu_9_layer_call_and_return_conditional_losses_69693752
re_lu_9/PartitionedCallÏ
IdentityIdentity re_lu_9/PartitionedCall:output:0.^batch_normalization_9/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬
ª
7__inference_batch_normalization_8_layer_call_fn_6975868

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69689432
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¼
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974734
conv2d_8_input+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOpÇ
conv2d_8/Conv2DConv2Dconv2d_8_input&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_8/BiasAdd¶
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOp¼
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1é
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3
re_lu_8/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_8/Reluv
IdentityIdentityre_lu_8/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:::::::_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_8_input
Ê
¯
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6969225

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Þ

S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976627

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974820
conv2d_9_input+
'conv2d_9_conv2d_readvariableop_resource,
(conv2d_9_biasadd_readvariableop_resource1
-batch_normalization_9_readvariableop_resource3
/batch_normalization_9_readvariableop_1_resourceB
>batch_normalization_9_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_9/Conv2D/ReadVariableOpÇ
conv2d_9/Conv2DConv2Dconv2d_9_input&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_9/Conv2D§
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_9/BiasAdd/ReadVariableOp¬
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_9/BiasAdd¶
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_9/ReadVariableOp¼
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_9/ReadVariableOp_1é
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_9/BiasAdd:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3
re_lu_9/ReluRelu*batch_normalization_9/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_9/Reluv
IdentityIdentityre_lu_9/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :::::::_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_9_input
å
ú
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6970400

inputs
conv2d_12_6970384
conv2d_12_6970386"
batch_normalization_12_6970389"
batch_normalization_12_6970391"
batch_normalization_12_6970393"
batch_normalization_12_6970395
identity¢.batch_normalization_12/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¯
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_6970384conv2d_12_6970386*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_conv2d_12_layer_call_and_return_conditional_losses_69702202#
!conv2d_12/StatefulPartitionedCallØ
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_12_6970389batch_normalization_12_6970391batch_normalization_12_6970393batch_normalization_12_6970395*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_697027320
.batch_normalization_12/StatefulPartitionedCall
re_lu_12/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

   E8 *N
fIRG
E__inference_re_lu_12_layer_call_and_return_conditional_losses_69703142
re_lu_12/PartitionedCallÒ
IdentityIdentity!re_lu_12/PartitionedCall:output:0/^batch_normalization_12/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¾
-__inference_cnn_block_0_layer_call_fn_6974510

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*8
config_proto(&

CPU

GPU2*0J

   E8 *Q
fLRJ
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69688352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ@tensorflow/serving/predict:ë
ì
feature_extractor
head
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"
_tf_keras_modelò{"class_name": "ContrastiveCNN", "name": "contrastive_cnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ContrastiveCNN"}}
È

cnn_blocks
	gba

	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_model{"class_name": "FeatureExtractorCNN", "name": "feature_extractor_cnn_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "FeatureExtractorCNN"}}
ê

kernel
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
î
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
/28
029
130
231
332
433
534
635
736
837
938
:39
;40
<41
42"
trackable_list_wrapper
 "
trackable_list_wrapper
þ
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
28"
trackable_list_wrapper
Î
	variables
=layer_metrics

>layers
?non_trainable_variables
regularization_losses
@metrics
Alayer_regularization_losses
trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
Q
B0
C1
D2
E3
F4
G5
H6"
trackable_list_wrapper

I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
+&call_and_return_all_conditional_losses
 __call__"
_tf_keras_layerî{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
æ
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
/28
029
130
231
332
433
534
635
736
837
938
:39
;40
<41"
trackable_list_wrapper
 "
trackable_list_wrapper
ö
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27"
trackable_list_wrapper
°

	variables
Mlayer_metrics

Nlayers
Onon_trainable_variables
regularization_losses
Pmetrics
Qlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
1:/	@2contrastive_cnn/dense_2/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
	variables
Rlayer_metrics

Slayers
Tnon_trainable_variables
regularization_losses
Umetrics
Vlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_7/kernel
:2conv2d_7/bias
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
):' 2conv2d_8/kernel
: 2conv2d_8/bias
):' 2batch_normalization_8/gamma
(:& 2batch_normalization_8/beta
):'  2conv2d_9/kernel
: 2conv2d_9/bias
):' 2batch_normalization_9/gamma
(:& 2batch_normalization_9/beta
*:(  2conv2d_10/kernel
: 2conv2d_10/bias
*:( 2batch_normalization_10/gamma
):' 2batch_normalization_10/beta
*:( @2conv2d_11/kernel
:@2conv2d_11/bias
*:(@2batch_normalization_11/gamma
):'@2batch_normalization_11/beta
*:(@@2conv2d_12/kernel
:@2conv2d_12/bias
*:(@2batch_normalization_12/gamma
):'@2batch_normalization_12/beta
+:)@2conv2d_13/kernel
:2conv2d_13/bias
+:)2batch_normalization_13/gamma
*:(2batch_normalization_13/beta
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
1:/  (2!batch_normalization_8/moving_mean
5:3  (2%batch_normalization_8/moving_variance
1:/  (2!batch_normalization_9/moving_mean
5:3  (2%batch_normalization_9/moving_variance
2:0  (2"batch_normalization_10/moving_mean
6:4  (2&batch_normalization_10/moving_variance
2:0@ (2"batch_normalization_11/moving_mean
6:4@ (2&batch_normalization_11/moving_variance
2:0@ (2"batch_normalization_12/moving_mean
6:4@ (2&batch_normalization_12/moving_variance
3:1 (2"batch_normalization_13/moving_mean
7:5 (2&batch_normalization_13/moving_variance
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper

/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ë
Wlayer_with_weights-0
Wlayer-0
Xlayer_with_weights-1
Xlayer-1
Ylayer-2
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+¡&call_and_return_all_conditional_losses
¢__call__"ÿ
_tf_keras_sequentialà{"class_name": "Sequential", "name": "cnn_block_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_7_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_7_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ï
^layer_with_weights-0
^layer-0
_layer_with_weights-1
_layer-1
`layer-2
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+£&call_and_return_all_conditional_losses
¤__call__"
_tf_keras_sequentialä{"class_name": "Sequential", "name": "cnn_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 26, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 26, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ï
elayer_with_weights-0
elayer-0
flayer_with_weights-1
flayer-1
glayer-2
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
+¥&call_and_return_all_conditional_losses
¦__call__"
_tf_keras_sequentialä{"class_name": "Sequential", "name": "cnn_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_9_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_9_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
÷
llayer_with_weights-0
llayer-0
mlayer_with_weights-1
mlayer-1
nlayer-2
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"
_tf_keras_sequentialì{"class_name": "Sequential", "name": "cnn_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22, 22, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22, 22, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
÷
slayer_with_weights-0
slayer-0
tlayer_with_weights-1
tlayer-1
ulayer-2
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
+©&call_and_return_all_conditional_losses
ª__call__"
_tf_keras_sequentialì{"class_name": "Sequential", "name": "cnn_block_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_11_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_11", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_11_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_11", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ø
zlayer_with_weights-0
zlayer-0
{layer_with_weights-1
{layer-1
|layer-2
}	variables
~regularization_losses
trainable_variables
	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"
_tf_keras_sequentialì{"class_name": "Sequential", "name": "cnn_block_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18, 18, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_12_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_12", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 18, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18, 18, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_12_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_12", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
 
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
+­&call_and_return_all_conditional_losses
®__call__"
_tf_keras_sequentialî{"class_name": "Sequential", "name": "cnn_block_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_13_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_13", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_13_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_13", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
I	variables
layer_metrics
layers
non_trainable_variables
Jregularization_losses
metrics
 layer_regularization_losses
Ktrainable_variables
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
X
B0
C1
D2
E3
F4
G5
H6
	7"
trackable_list_wrapper

/0
01
12
23
34
45
56
67
78
89
910
:11
;12
<13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡

_inbound_nodes

kernel
bias
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+¯&call_and_return_all_conditional_losses
°__call__"Ë
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 28, 28, 1]}}
é	
_inbound_nodes
	axis
	gamma
beta
/moving_mean
0moving_variance
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+±&call_and_return_all_conditional_losses
²__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 26, 26, 16]}}

_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+³&call_and_return_all_conditional_losses
´__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
0
1
2
3
/4
05"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
µ
Z	variables
layer_metrics
 layers
¡non_trainable_variables
[regularization_losses
¢metrics
 £layer_regularization_losses
\trainable_variables
¢__call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
£

¤_inbound_nodes

kernel
bias
¥_outbound_nodes
¦	variables
§regularization_losses
¨trainable_variables
©	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 26, 26, 16]}}
é	
ª_inbound_nodes
	«axis
	gamma
beta
1moving_mean
2moving_variance
¬_outbound_nodes
­	variables
®regularization_losses
¯trainable_variables
°	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 24, 24, 32]}}

±_inbound_nodes
²	variables
³regularization_losses
´trainable_variables
µ	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
0
1
2
3
14
25"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
µ
a	variables
¶layer_metrics
·layers
¸non_trainable_variables
bregularization_losses
¹metrics
 ºlayer_regularization_losses
ctrainable_variables
¤__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
£

»_inbound_nodes

kernel
bias
¼_outbound_nodes
½	variables
¾regularization_losses
¿trainable_variables
À	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 24, 24, 32]}}
é	
Á_inbound_nodes
	Âaxis
	gamma
beta
3moving_mean
4moving_variance
Ã_outbound_nodes
Ä	variables
Åregularization_losses
Ætrainable_variables
Ç	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 22, 22, 32]}}

È_inbound_nodes
É	variables
Êregularization_losses
Ëtrainable_variables
Ì	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_9", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
0
1
2
3
34
45"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
µ
h	variables
Ílayer_metrics
Îlayers
Ïnon_trainable_variables
iregularization_losses
Ðmetrics
 Ñlayer_regularization_losses
jtrainable_variables
¦__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
¥

Ò_inbound_nodes

kernel
 bias
Ó_outbound_nodes
Ô	variables
Õregularization_losses
Ötrainable_variables
×	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 22, 22, 32]}}
ë	
Ø_inbound_nodes
	Ùaxis
	!gamma
"beta
5moving_mean
6moving_variance
Ú_outbound_nodes
Û	variables
Üregularization_losses
Ýtrainable_variables
Þ	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 20, 20, 32]}}

ß_inbound_nodes
à	variables
áregularization_losses
âtrainable_variables
ã	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"Þ
_tf_keras_layerÄ{"class_name": "ReLU", "name": "re_lu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_10", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
0
 1
!2
"3
54
65"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
µ
o	variables
älayer_metrics
ålayers
ænon_trainable_variables
pregularization_losses
çmetrics
 èlayer_regularization_losses
qtrainable_variables
¨__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
¥

é_inbound_nodes

#kernel
$bias
ê_outbound_nodes
ë	variables
ìregularization_losses
ítrainable_variables
î	keras_api
+Ç&call_and_return_all_conditional_losses
È__call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 20, 20, 32]}}
ë	
ï_inbound_nodes
	ðaxis
	%gamma
&beta
7moving_mean
8moving_variance
ñ_outbound_nodes
ò	variables
óregularization_losses
ôtrainable_variables
õ	keras_api
+É&call_and_return_all_conditional_losses
Ê__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 18, 18, 64]}}

ö_inbound_nodes
÷	variables
øregularization_losses
ùtrainable_variables
ú	keras_api
+Ë&call_and_return_all_conditional_losses
Ì__call__"Þ
_tf_keras_layerÄ{"class_name": "ReLU", "name": "re_lu_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_11", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
#0
$1
%2
&3
74
85"
trackable_list_wrapper
 "
trackable_list_wrapper
<
#0
$1
%2
&3"
trackable_list_wrapper
µ
v	variables
ûlayer_metrics
ülayers
ýnon_trainable_variables
wregularization_losses
þmetrics
 ÿlayer_regularization_losses
xtrainable_variables
ª__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
¥

_inbound_nodes

'kernel
(bias
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+Í&call_and_return_all_conditional_losses
Î__call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 18, 18, 64]}}
ë	
_inbound_nodes
	axis
	)gamma
*beta
9moving_mean
:moving_variance
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+Ï&call_and_return_all_conditional_losses
Ð__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_12", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16, 16, 64]}}

_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+Ñ&call_and_return_all_conditional_losses
Ò__call__"Þ
_tf_keras_layerÄ{"class_name": "ReLU", "name": "re_lu_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_12", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
'0
(1
)2
*3
94
:5"
trackable_list_wrapper
 "
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
µ
}	variables
layer_metrics
layers
non_trainable_variables
~regularization_losses
metrics
 layer_regularization_losses
trainable_variables
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
¦

_inbound_nodes

+kernel
,bias
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+Ó&call_and_return_all_conditional_losses
Ô__call__"Ð
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16, 16, 64]}}
í	
_inbound_nodes
	axis
	-gamma
.beta
;moving_mean
<moving_variance
_outbound_nodes
 	variables
¡regularization_losses
¢trainable_variables
£	keras_api
+Õ&call_and_return_all_conditional_losses
Ö__call__"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_13", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 14, 14, 128]}}

¤_inbound_nodes
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
+×&call_and_return_all_conditional_losses
Ø__call__"Þ
_tf_keras_layerÄ{"class_name": "ReLU", "name": "re_lu_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_13", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
+0
,1
-2
.3
;4
<5"
trackable_list_wrapper
 "
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
¸
	variables
©layer_metrics
ªlayers
«non_trainable_variables
regularization_losses
¬metrics
 ­layer_regularization_losses
trainable_variables
®__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
	variables
®layer_metrics
¯layers
°non_trainable_variables
regularization_losses
±metrics
 ²layer_regularization_losses
trainable_variables
°__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
/2
03"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
	variables
³layer_metrics
´layers
µnon_trainable_variables
regularization_losses
¶metrics
 ·layer_regularization_losses
trainable_variables
²__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
¸layer_metrics
¹layers
ºnon_trainable_variables
regularization_losses
»metrics
 ¼layer_regularization_losses
trainable_variables
´__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
W0
X1
Y2"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
¦	variables
½layer_metrics
¾layers
¿non_trainable_variables
§regularization_losses
Àmetrics
 Álayer_regularization_losses
¨trainable_variables
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
­	variables
Âlayer_metrics
Ãlayers
Änon_trainable_variables
®regularization_losses
Åmetrics
 Ælayer_regularization_losses
¯trainable_variables
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
²	variables
Çlayer_metrics
Èlayers
Énon_trainable_variables
³regularization_losses
Êmetrics
 Ëlayer_regularization_losses
´trainable_variables
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
^0
_1
`2"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
½	variables
Ìlayer_metrics
Ílayers
Înon_trainable_variables
¾regularization_losses
Ïmetrics
 Ðlayer_regularization_losses
¿trainable_variables
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
32
43"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
Ä	variables
Ñlayer_metrics
Òlayers
Ónon_trainable_variables
Åregularization_losses
Ômetrics
 Õlayer_regularization_losses
Ætrainable_variables
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
É	variables
Ölayer_metrics
×layers
Ønon_trainable_variables
Êregularization_losses
Ùmetrics
 Úlayer_regularization_losses
Ëtrainable_variables
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
e0
f1
g2"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
¸
Ô	variables
Ûlayer_metrics
Ülayers
Ýnon_trainable_variables
Õregularization_losses
Þmetrics
 ßlayer_regularization_losses
Ötrainable_variables
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
!0
"1
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
¸
Û	variables
àlayer_metrics
álayers
ânon_trainable_variables
Üregularization_losses
ãmetrics
 älayer_regularization_losses
Ýtrainable_variables
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
à	variables
ålayer_metrics
ælayers
çnon_trainable_variables
áregularization_losses
èmetrics
 élayer_regularization_losses
âtrainable_variables
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
l0
m1
n2"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
¸
ë	variables
êlayer_metrics
ëlayers
ìnon_trainable_variables
ìregularization_losses
ímetrics
 îlayer_regularization_losses
ítrainable_variables
È__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
%0
&1
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
¸
ò	variables
ïlayer_metrics
ðlayers
ñnon_trainable_variables
óregularization_losses
òmetrics
 ólayer_regularization_losses
ôtrainable_variables
Ê__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
÷	variables
ôlayer_metrics
õlayers
önon_trainable_variables
øregularization_losses
÷metrics
 ølayer_regularization_losses
ùtrainable_variables
Ì__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
s0
t1
u2"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
¸
	variables
ùlayer_metrics
úlayers
ûnon_trainable_variables
regularization_losses
ümetrics
 ýlayer_regularization_losses
trainable_variables
Î__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
)0
*1
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
¸
	variables
þlayer_metrics
ÿlayers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
Ð__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
Ò__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
z0
{1
|2"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
¸
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
Ô__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
-0
.1
;2
<3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
¸
 	variables
layer_metrics
layers
non_trainable_variables
¡regularization_losses
metrics
 layer_regularization_losses
¢trainable_variables
Ö__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¥	variables
layer_metrics
layers
non_trainable_variables
¦regularization_losses
metrics
 layer_regularization_losses
§trainable_variables
Ø__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
8
0
1
2"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ò2ï
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6973074
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6972562
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6972720
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6973232´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
1__inference_contrastive_cnn_layer_call_fn_6973414
1__inference_contrastive_cnn_layer_call_fn_6972902
1__inference_contrastive_cnn_layer_call_fn_6973323
1__inference_contrastive_cnn_layer_call_fn_6972811´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
è2å
"__inference__wrapped_model_6968537¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
2
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6974232
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6973734
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6973581
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6974079°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
9__inference_feature_extractor_cnn_1_layer_call_fn_6974321
9__inference_feature_extractor_cnn_1_layer_call_fn_6973912
9__inference_feature_extractor_cnn_1_layer_call_fn_6974410
9__inference_feature_extractor_cnn_1_layer_call_fn_6973823°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_dense_2_layer_call_and_return_conditional_losses_6974417¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_2_layer_call_fn_6974424¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
4B2
%__inference_signature_wrapper_6972390input_1
¿2¼
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_6970735à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¤2¡
<__inference_global_average_pooling2d_1_layer_call_fn_6970741à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
î2ë
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974537
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974476
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974451
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974562À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_cnn_block_0_layer_call_fn_6974596
-__inference_cnn_block_0_layer_call_fn_6974493
-__inference_cnn_block_0_layer_call_fn_6974510
-__inference_cnn_block_0_layer_call_fn_6974579À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974623
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974648
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974734
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974709À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_cnn_block_1_layer_call_fn_6974682
-__inference_cnn_block_1_layer_call_fn_6974768
-__inference_cnn_block_1_layer_call_fn_6974665
-__inference_cnn_block_1_layer_call_fn_6974751À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974881
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974906
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974795
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974820À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_cnn_block_2_layer_call_fn_6974837
-__inference_cnn_block_2_layer_call_fn_6974940
-__inference_cnn_block_2_layer_call_fn_6974923
-__inference_cnn_block_2_layer_call_fn_6974854À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6974992
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6974967
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6975078
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6975053À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_cnn_block_3_layer_call_fn_6975009
-__inference_cnn_block_3_layer_call_fn_6975095
-__inference_cnn_block_3_layer_call_fn_6975026
-__inference_cnn_block_3_layer_call_fn_6975112À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975250
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975225
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975139
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975164À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_cnn_block_4_layer_call_fn_6975267
-__inference_cnn_block_4_layer_call_fn_6975181
-__inference_cnn_block_4_layer_call_fn_6975284
-__inference_cnn_block_4_layer_call_fn_6975198À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975311
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975336
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975422
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975397À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_cnn_block_5_layer_call_fn_6975370
-__inference_cnn_block_5_layer_call_fn_6975439
-__inference_cnn_block_5_layer_call_fn_6975456
-__inference_cnn_block_5_layer_call_fn_6975353À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975483
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975569
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975508
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975594À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_cnn_block_6_layer_call_fn_6975525
-__inference_cnn_block_6_layer_call_fn_6975628
-__inference_cnn_block_6_layer_call_fn_6975542
-__inference_cnn_block_6_layer_call_fn_6975611À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6975638¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_7_layer_call_fn_6975647¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975685
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975667
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975731
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975749´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_7_layer_call_fn_6975711
7__inference_batch_normalization_7_layer_call_fn_6975775
7__inference_batch_normalization_7_layer_call_fn_6975762
7__inference_batch_normalization_7_layer_call_fn_6975698´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6975780¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_re_lu_7_layer_call_fn_6975785¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6975795¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_8_layer_call_fn_6975804¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975888
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975906
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975842
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975824´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_8_layer_call_fn_6975932
7__inference_batch_normalization_8_layer_call_fn_6975868
7__inference_batch_normalization_8_layer_call_fn_6975919
7__inference_batch_normalization_8_layer_call_fn_6975855´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6975937¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_re_lu_8_layer_call_fn_6975942¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6975952¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_9_layer_call_fn_6975961¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6975981
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6976063
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6976045
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6975999´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_9_layer_call_fn_6976089
7__inference_batch_normalization_9_layer_call_fn_6976025
7__inference_batch_normalization_9_layer_call_fn_6976012
7__inference_batch_normalization_9_layer_call_fn_6976076´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_re_lu_9_layer_call_and_return_conditional_losses_6976094¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_re_lu_9_layer_call_fn_6976099¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_10_layer_call_and_return_conditional_losses_6976109¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv2d_10_layer_call_fn_6976118¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976156
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976220
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976138
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976202´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
8__inference_batch_normalization_10_layer_call_fn_6976246
8__inference_batch_normalization_10_layer_call_fn_6976182
8__inference_batch_normalization_10_layer_call_fn_6976233
8__inference_batch_normalization_10_layer_call_fn_6976169´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_re_lu_10_layer_call_and_return_conditional_losses_6976251¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_re_lu_10_layer_call_fn_6976256¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_11_layer_call_and_return_conditional_losses_6976266¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv2d_11_layer_call_fn_6976275¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976359
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976295
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976313
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976377´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
8__inference_batch_normalization_11_layer_call_fn_6976339
8__inference_batch_normalization_11_layer_call_fn_6976390
8__inference_batch_normalization_11_layer_call_fn_6976403
8__inference_batch_normalization_11_layer_call_fn_6976326´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_re_lu_11_layer_call_and_return_conditional_losses_6976408¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_re_lu_11_layer_call_fn_6976413¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_12_layer_call_and_return_conditional_losses_6976423¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv2d_12_layer_call_fn_6976432¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976470
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976452
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976516
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976534´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
8__inference_batch_normalization_12_layer_call_fn_6976547
8__inference_batch_normalization_12_layer_call_fn_6976496
8__inference_batch_normalization_12_layer_call_fn_6976483
8__inference_batch_normalization_12_layer_call_fn_6976560´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_re_lu_12_layer_call_and_return_conditional_losses_6976565¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_re_lu_12_layer_call_fn_6976570¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_13_layer_call_and_return_conditional_losses_6976580¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv2d_13_layer_call_fn_6976589¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976691
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976627
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976673
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976609´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
8__inference_batch_normalization_13_layer_call_fn_6976717
8__inference_batch_normalization_13_layer_call_fn_6976704
8__inference_batch_normalization_13_layer_call_fn_6976653
8__inference_batch_normalization_13_layer_call_fn_6976640´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_re_lu_13_layer_call_and_return_conditional_losses_6976722¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_re_lu_13_layer_call_fn_6976727¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ã
"__inference__wrapped_model_6968537+/01234 !"56#$%&78'()*9:+,-.;<8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ@î
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976138!"56M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 î
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976156!"56M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 É
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976202r!"56;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 É
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_6976220r!"56;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Æ
8__inference_batch_normalization_10_layer_call_fn_6976169!"56M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Æ
8__inference_batch_normalization_10_layer_call_fn_6976182!"56M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¡
8__inference_batch_normalization_10_layer_call_fn_6976233e!"56;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ ¡
8__inference_batch_normalization_10_layer_call_fn_6976246e!"56;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ î
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976295%&78M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 î
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976313%&78M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 É
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976359r%&78;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 É
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_6976377r%&78;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Æ
8__inference_batch_normalization_11_layer_call_fn_6976326%&78M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Æ
8__inference_batch_normalization_11_layer_call_fn_6976339%&78M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¡
8__inference_batch_normalization_11_layer_call_fn_6976390e%&78;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@¡
8__inference_batch_normalization_11_layer_call_fn_6976403e%&78;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@î
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976452)*9:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 î
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976470)*9:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 É
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976516r)*9:;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 É
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_6976534r)*9:;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Æ
8__inference_batch_normalization_12_layer_call_fn_6976483)*9:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Æ
8__inference_batch_normalization_12_layer_call_fn_6976496)*9:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¡
8__inference_batch_normalization_12_layer_call_fn_6976547e)*9:;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@¡
8__inference_batch_normalization_12_layer_call_fn_6976560e)*9:;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@Ë
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976609t-.;<<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ë
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976627t-.;<<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976673-.;<N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_6976691-.;<N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 £
8__inference_batch_normalization_13_layer_call_fn_6976640g-.;<<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ£
8__inference_batch_normalization_13_layer_call_fn_6976653g-.;<<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_13_layer_call_fn_6976704-.;<N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_13_layer_call_fn_6976717-.;<N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975667/0M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 í
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975685/0M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975731r/0;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 È
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6975749r/0;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Å
7__inference_batch_normalization_7_layer_call_fn_6975698/0M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
7__inference_batch_normalization_7_layer_call_fn_6975711/0M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
7__inference_batch_normalization_7_layer_call_fn_6975762e/0;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ 
7__inference_batch_normalization_7_layer_call_fn_6975775e/0;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿí
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_697582412M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 í
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_697584212M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975888r12;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_6975906r12;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Å
7__inference_batch_normalization_8_layer_call_fn_697585512M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Å
7__inference_batch_normalization_8_layer_call_fn_697586812M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_8_layer_call_fn_6975919e12;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_8_layer_call_fn_6975932e12;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ È
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6975981r34;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_6975999r34;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 í
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_697604534M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 í
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_697606334M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
  
7__inference_batch_normalization_9_layer_call_fn_6976012e34;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_9_layer_call_fn_6976025e34;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ Å
7__inference_batch_normalization_9_layer_call_fn_697607634M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Å
7__inference_batch_normalization_9_layer_call_fn_697608934M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ä
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974451x/0?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974476x/0?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Í
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974537/0G¢D
=¢:
0-
conv2d_7_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Í
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6974562/0G¢D
=¢:
0-
conv2d_7_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_cnn_block_0_layer_call_fn_6974493k/0?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ
-__inference_cnn_block_0_layer_call_fn_6974510k/0?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ¤
-__inference_cnn_block_0_layer_call_fn_6974579s/0G¢D
=¢:
0-
conv2d_7_inputÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ¤
-__inference_cnn_block_0_layer_call_fn_6974596s/0G¢D
=¢:
0-
conv2d_7_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿÄ
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974623x12?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ä
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6974648x12?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Í
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_697470912G¢D
=¢:
0-
conv2d_8_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Í
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_697473412G¢D
=¢:
0-
conv2d_8_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
-__inference_cnn_block_1_layer_call_fn_6974665k12?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_1_layer_call_fn_6974682k12?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ ¤
-__inference_cnn_block_1_layer_call_fn_6974751s12G¢D
=¢:
0-
conv2d_8_inputÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ ¤
-__inference_cnn_block_1_layer_call_fn_6974768s12G¢D
=¢:
0-
conv2d_8_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ Í
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_697479534G¢D
=¢:
0-
conv2d_9_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Í
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_697482034G¢D
=¢:
0-
conv2d_9_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ä
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974881x34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ä
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6974906x34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ¤
-__inference_cnn_block_2_layer_call_fn_6974837s34G¢D
=¢:
0-
conv2d_9_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ ¤
-__inference_cnn_block_2_layer_call_fn_6974854s34G¢D
=¢:
0-
conv2d_9_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_2_layer_call_fn_6974923k34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_2_layer_call_fn_6974940k34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ Ä
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6974967x !"56?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ä
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6974992x !"56?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Î
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6975053 !"56H¢E
>¢;
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Î
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6975078 !"56H¢E
>¢;
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
-__inference_cnn_block_3_layer_call_fn_6975009k !"56?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_3_layer_call_fn_6975026k !"56?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ ¥
-__inference_cnn_block_3_layer_call_fn_6975095t !"56H¢E
>¢;
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ ¥
-__inference_cnn_block_3_layer_call_fn_6975112t !"56H¢E
>¢;
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ Ä
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975139x#$%&78?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ä
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975164x#$%&78?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Î
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975225#$%&78H¢E
>¢;
1.
conv2d_11_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Î
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6975250#$%&78H¢E
>¢;
1.
conv2d_11_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_cnn_block_4_layer_call_fn_6975181k#$%&78?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ@
-__inference_cnn_block_4_layer_call_fn_6975198k#$%&78?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@¥
-__inference_cnn_block_4_layer_call_fn_6975267t#$%&78H¢E
>¢;
1.
conv2d_11_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ@¥
-__inference_cnn_block_4_layer_call_fn_6975284t#$%&78H¢E
>¢;
1.
conv2d_11_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@Ä
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975311x'()*9:?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ä
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975336x'()*9:?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Î
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975397'()*9:H¢E
>¢;
1.
conv2d_12_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Î
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6975422'()*9:H¢E
>¢;
1.
conv2d_12_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_cnn_block_5_layer_call_fn_6975353k'()*9:?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª " ÿÿÿÿÿÿÿÿÿ@
-__inference_cnn_block_5_layer_call_fn_6975370k'()*9:?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@¥
-__inference_cnn_block_5_layer_call_fn_6975439t'()*9:H¢E
>¢;
1.
conv2d_12_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª " ÿÿÿÿÿÿÿÿÿ@¥
-__inference_cnn_block_5_layer_call_fn_6975456t'()*9:H¢E
>¢;
1.
conv2d_12_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@Ï
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975483+,-.;<H¢E
>¢;
1.
conv2d_13_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ï
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975508+,-.;<H¢E
>¢;
1.
conv2d_13_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Å
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975569y+,-.;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Å
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6975594y+,-.;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¦
-__inference_cnn_block_6_layer_call_fn_6975525u+,-.;<H¢E
>¢;
1.
conv2d_13_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ¦
-__inference_cnn_block_6_layer_call_fn_6975542u+,-.;<H¢E
>¢;
1.
conv2d_13_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ
-__inference_cnn_block_6_layer_call_fn_6975611l+,-.;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ
-__inference_cnn_block_6_layer_call_fn_6975628l+,-.;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿã
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6972562+/01234 !"56#$%&78'()*9:+,-.;<<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ã
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6972720+/01234 !"56#$%&78'()*9:+,-.;<<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 â
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6973074+/01234 !"56#$%&78'()*9:+,-.;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 â
L__inference_contrastive_cnn_layer_call_and_return_conditional_losses_6973232+/01234 !"56#$%&78'()*9:+,-.;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 »
1__inference_contrastive_cnn_layer_call_fn_6972811+/01234 !"56#$%&78'()*9:+,-.;<<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ@»
1__inference_contrastive_cnn_layer_call_fn_6972902+/01234 !"56#$%&78'()*9:+,-.;<<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ@º
1__inference_contrastive_cnn_layer_call_fn_6973323+/01234 !"56#$%&78'()*9:+,-.;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ@º
1__inference_contrastive_cnn_layer_call_fn_6973414+/01234 !"56#$%&78'()*9:+,-.;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ@¶
F__inference_conv2d_10_layer_call_and_return_conditional_losses_6976109l 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
+__inference_conv2d_10_layer_call_fn_6976118_ 7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ¶
F__inference_conv2d_11_layer_call_and_return_conditional_losses_6976266l#$7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv2d_11_layer_call_fn_6976275_#$7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@¶
F__inference_conv2d_12_layer_call_and_return_conditional_losses_6976423l'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv2d_12_layer_call_fn_6976432_'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@·
F__inference_conv2d_13_layer_call_and_return_conditional_losses_6976580m+,7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv2d_13_layer_call_fn_6976589`+,7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿµ
E__inference_conv2d_7_layer_call_and_return_conditional_losses_6975638l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_7_layer_call_fn_6975647_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿµ
E__inference_conv2d_8_layer_call_and_return_conditional_losses_6975795l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv2d_8_layer_call_fn_6975804_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ µ
E__inference_conv2d_9_layer_call_and_return_conditional_losses_6975952l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv2d_9_layer_call_fn_6975961_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ¤
D__inference_dense_2_layer_call_and_return_conditional_losses_6974417\0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dense_2_layer_call_fn_6974424O0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@ê
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6973581*/01234 !"56#$%&78'()*9:+,-.;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ê
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6973734*/01234 !"56#$%&78'()*9:+,-.;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ë
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6974079*/01234 !"56#$%&78'()*9:+,-.;<<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ë
T__inference_feature_extractor_cnn_1_layer_call_and_return_conditional_losses_6974232*/01234 !"56#$%&78'()*9:+,-.;<<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Â
9__inference_feature_extractor_cnn_1_layer_call_fn_6973823*/01234 !"56#$%&78'()*9:+,-.;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÂ
9__inference_feature_extractor_cnn_1_layer_call_fn_6973912*/01234 !"56#$%&78'()*9:+,-.;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿÃ
9__inference_feature_extractor_cnn_1_layer_call_fn_6974321*/01234 !"56#$%&78'()*9:+,-.;<<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÃ
9__inference_feature_extractor_cnn_1_layer_call_fn_6974410*/01234 !"56#$%&78'()*9:+,-.;<<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿà
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_6970735R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ·
<__inference_global_average_pooling2d_1_layer_call_fn_6970741wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
E__inference_re_lu_10_layer_call_and_return_conditional_losses_6976251h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_re_lu_10_layer_call_fn_6976256[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ ±
E__inference_re_lu_11_layer_call_and_return_conditional_losses_6976408h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_re_lu_11_layer_call_fn_6976413[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@±
E__inference_re_lu_12_layer_call_and_return_conditional_losses_6976565h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_re_lu_12_layer_call_fn_6976570[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@³
E__inference_re_lu_13_layer_call_and_return_conditional_losses_6976722j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_re_lu_13_layer_call_fn_6976727]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ°
D__inference_re_lu_7_layer_call_and_return_conditional_losses_6975780h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_re_lu_7_layer_call_fn_6975785[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ°
D__inference_re_lu_8_layer_call_and_return_conditional_losses_6975937h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_re_lu_8_layer_call_fn_6975942[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ °
D__inference_re_lu_9_layer_call_and_return_conditional_losses_6976094h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_re_lu_9_layer_call_fn_6976099[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ Ñ
%__inference_signature_wrapper_6972390§+/01234 !"56#$%&78'()*9:+,-.;<C¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ@