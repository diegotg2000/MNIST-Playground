Úú7
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
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ú/
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0

batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_6/kernel
|
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*'
_output_shapes
:@*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:*
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:*
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
¢
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
¢
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
¨
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*â
value×BÓ BË
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
­
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
æ
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
)20
*21
+22
,23
-24
.25
/26
027
128
229
330
431
532
633
734
835
936
:37
;38
<39
=40
>41
?42
@43
A44
B45
 
ö
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
)20
*21
+22
,23
-24
.25
/26
027
?28
@29
A30
B31
­
	variables
Clayer_metrics

Dlayers
Enon_trainable_variables
regularization_losses
Fmetrics
Glayer_regularization_losses
trainable_variables
 
1
H0
I1
J2
K3
L4
M5
N6
R
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
Æ
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
)20
*21
+22
,23
-24
.25
/26
027
128
229
330
431
532
633
734
835
936
:37
;38
<39
=40
>41
 
Ö
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
)20
*21
+22
,23
-24
.25
/26
027
­

	variables
Slayer_metrics

Tlayers
Unon_trainable_variables
regularization_losses
Vmetrics
Wlayer_regularization_losses
trainable_variables

X_inbound_nodes

?kernel
@bias
Y_outbound_nodes
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
{
^_inbound_nodes
__outbound_nodes
`	variables
aregularization_losses
btrainable_variables
c	keras_api
|
d_inbound_nodes

Akernel
Bbias
e	variables
fregularization_losses
gtrainable_variables
h	keras_api

?0
@1
A2
B3
 

?0
@1
A2
B3
­
	variables
ilayer_metrics

jlayers
knon_trainable_variables
regularization_losses
lmetrics
mlayer_regularization_losses
trainable_variables
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_3/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_3/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_4/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_4/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_4/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_4/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_5/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_5/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_5/gamma'variables/22/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_5/beta'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_6/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_6/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_6/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_6/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEbatch_normalization/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#batch_normalization/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/30/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/31/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/32/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/33/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_4/moving_mean'variables/36/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_4/moving_variance'variables/37/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_5/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_5/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_6/moving_mean'variables/40/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_6/moving_variance'variables/41/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUE
dense/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
f
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
 
 
­
nlayer_with_weights-0
nlayer-0
olayer_with_weights-1
olayer-1
player-2
q	variables
rregularization_losses
strainable_variables
t	keras_api
­
ulayer_with_weights-0
ulayer-0
vlayer_with_weights-1
vlayer-1
wlayer-2
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
°
|layer_with_weights-0
|layer-0
}layer_with_weights-1
}layer-1
~layer-2
	variables
regularization_losses
trainable_variables
	keras_api
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
 
 
 
²
O	variables
layer_metrics
 layers
¡non_trainable_variables
Pregularization_losses
¢metrics
 £layer_regularization_losses
Qtrainable_variables
 
8
H0
I1
J2
K3
L4
M5
N6
	7
f
10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13
 
 
 
 

?0
@1
 

?0
@1
²
Z	variables
¤layer_metrics
¥layers
¦non_trainable_variables
[regularization_losses
§metrics
 ¨layer_regularization_losses
\trainable_variables
 
 
 
 
 
²
`	variables
©layer_metrics
ªlayers
«non_trainable_variables
aregularization_losses
¬metrics
 ­layer_regularization_losses
btrainable_variables
 

A0
B1
 

A0
B1
²
e	variables
®layer_metrics
¯layers
°non_trainable_variables
fregularization_losses
±metrics
 ²layer_regularization_losses
gtrainable_variables
 

0
1
2
 
 
 

³_inbound_nodes

kernel
bias
´_outbound_nodes
µ	variables
¶regularization_losses
·trainable_variables
¸	keras_api
Ç
¹_inbound_nodes
	ºaxis
	gamma
beta
1moving_mean
2moving_variance
»_outbound_nodes
¼	variables
½regularization_losses
¾trainable_variables
¿	keras_api
k
À_inbound_nodes
Á	variables
Âregularization_losses
Ãtrainable_variables
Ä	keras_api
*
0
1
2
3
14
25
 

0
1
2
3
²
q	variables
Ålayer_metrics
Ælayers
Çnon_trainable_variables
rregularization_losses
Èmetrics
 Élayer_regularization_losses
strainable_variables

Ê_inbound_nodes

kernel
bias
Ë_outbound_nodes
Ì	variables
Íregularization_losses
Îtrainable_variables
Ï	keras_api
Ç
Ð_inbound_nodes
	Ñaxis
	gamma
beta
3moving_mean
4moving_variance
Ò_outbound_nodes
Ó	variables
Ôregularization_losses
Õtrainable_variables
Ö	keras_api
k
×_inbound_nodes
Ø	variables
Ùregularization_losses
Útrainable_variables
Û	keras_api
*
0
1
2
3
34
45
 

0
1
2
3
²
x	variables
Ülayer_metrics
Ýlayers
Þnon_trainable_variables
yregularization_losses
ßmetrics
 àlayer_regularization_losses
ztrainable_variables

á_inbound_nodes

kernel
bias
â_outbound_nodes
ã	variables
äregularization_losses
åtrainable_variables
æ	keras_api
Ç
ç_inbound_nodes
	èaxis
	gamma
 beta
5moving_mean
6moving_variance
é_outbound_nodes
ê	variables
ëregularization_losses
ìtrainable_variables
í	keras_api
k
î_inbound_nodes
ï	variables
ðregularization_losses
ñtrainable_variables
ò	keras_api
*
0
1
2
 3
54
65
 

0
1
2
 3
´
	variables
ólayer_metrics
ôlayers
õnon_trainable_variables
regularization_losses
ömetrics
 ÷layer_regularization_losses
trainable_variables

ø_inbound_nodes

!kernel
"bias
ù_outbound_nodes
ú	variables
ûregularization_losses
ütrainable_variables
ý	keras_api
Ç
þ_inbound_nodes
	ÿaxis
	#gamma
$beta
7moving_mean
8moving_variance
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
k
_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
*
!0
"1
#2
$3
74
85
 

!0
"1
#2
$3
µ
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables

_inbound_nodes

%kernel
&bias
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
Ç
_inbound_nodes
	axis
	'gamma
(beta
9moving_mean
:moving_variance
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
k
_inbound_nodes
	variables
regularization_losses
trainable_variables
 	keras_api
*
%0
&1
'2
(3
94
:5
 

%0
&1
'2
(3
µ
	variables
¡layer_metrics
¢layers
£non_trainable_variables
regularization_losses
¤metrics
 ¥layer_regularization_losses
trainable_variables

¦_inbound_nodes

)kernel
*bias
§_outbound_nodes
¨	variables
©regularization_losses
ªtrainable_variables
«	keras_api
Ç
¬_inbound_nodes
	­axis
	+gamma
,beta
;moving_mean
<moving_variance
®_outbound_nodes
¯	variables
°regularization_losses
±trainable_variables
²	keras_api
k
³_inbound_nodes
´	variables
µregularization_losses
¶trainable_variables
·	keras_api
*
)0
*1
+2
,3
;4
<5
 

)0
*1
+2
,3
µ
	variables
¸layer_metrics
¹layers
ºnon_trainable_variables
regularization_losses
»metrics
 ¼layer_regularization_losses
trainable_variables

½_inbound_nodes

-kernel
.bias
¾_outbound_nodes
¿	variables
Àregularization_losses
Átrainable_variables
Â	keras_api
Ç
Ã_inbound_nodes
	Äaxis
	/gamma
0beta
=moving_mean
>moving_variance
Å_outbound_nodes
Æ	variables
Çregularization_losses
Ètrainable_variables
É	keras_api
k
Ê_inbound_nodes
Ë	variables
Ìregularization_losses
Ítrainable_variables
Î	keras_api
*
-0
.1
/2
03
=4
>5
 

-0
.1
/2
03
µ
	variables
Ïlayer_metrics
Ðlayers
Ñnon_trainable_variables
regularization_losses
Òmetrics
 Ólayer_regularization_losses
trainable_variables
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
 
 
 
 
 
 
 
 

0
1
 

0
1
µ
µ	variables
Ôlayer_metrics
Õlayers
Önon_trainable_variables
¶regularization_losses
×metrics
 Ølayer_regularization_losses
·trainable_variables
 
 
 

0
1
12
23
 

0
1
µ
¼	variables
Ùlayer_metrics
Úlayers
Ûnon_trainable_variables
½regularization_losses
Ümetrics
 Ýlayer_regularization_losses
¾trainable_variables
 
 
 
 
µ
Á	variables
Þlayer_metrics
ßlayers
ànon_trainable_variables
Âregularization_losses
ámetrics
 âlayer_regularization_losses
Ãtrainable_variables
 

n0
o1
p2

10
21
 
 
 
 

0
1
 

0
1
µ
Ì	variables
ãlayer_metrics
älayers
ånon_trainable_variables
Íregularization_losses
æmetrics
 çlayer_regularization_losses
Îtrainable_variables
 
 
 

0
1
32
43
 

0
1
µ
Ó	variables
èlayer_metrics
élayers
ênon_trainable_variables
Ôregularization_losses
ëmetrics
 ìlayer_regularization_losses
Õtrainable_variables
 
 
 
 
µ
Ø	variables
ílayer_metrics
îlayers
ïnon_trainable_variables
Ùregularization_losses
ðmetrics
 ñlayer_regularization_losses
Útrainable_variables
 

u0
v1
w2

30
41
 
 
 
 

0
1
 

0
1
µ
ã	variables
òlayer_metrics
ólayers
ônon_trainable_variables
äregularization_losses
õmetrics
 ölayer_regularization_losses
åtrainable_variables
 
 
 

0
 1
52
63
 

0
 1
µ
ê	variables
÷layer_metrics
ølayers
ùnon_trainable_variables
ëregularization_losses
úmetrics
 ûlayer_regularization_losses
ìtrainable_variables
 
 
 
 
µ
ï	variables
ülayer_metrics
ýlayers
þnon_trainable_variables
ðregularization_losses
ÿmetrics
 layer_regularization_losses
ñtrainable_variables
 

|0
}1
~2

50
61
 
 
 
 

!0
"1
 

!0
"1
µ
ú	variables
layer_metrics
layers
non_trainable_variables
ûregularization_losses
metrics
 layer_regularization_losses
ütrainable_variables
 
 
 

#0
$1
72
83
 

#0
$1
µ
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
 
 
 
 
µ
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
 

0
1
2

70
81
 
 
 
 

%0
&1
 

%0
&1
µ
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
 
 
 

'0
(1
92
:3
 

'0
(1
µ
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
 
 
 
 
µ
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
 

0
1
2

90
:1
 
 
 
 

)0
*1
 

)0
*1
µ
¨	variables
layer_metrics
 layers
¡non_trainable_variables
©regularization_losses
¢metrics
 £layer_regularization_losses
ªtrainable_variables
 
 
 

+0
,1
;2
<3
 

+0
,1
µ
¯	variables
¤layer_metrics
¥layers
¦non_trainable_variables
°regularization_losses
§metrics
 ¨layer_regularization_losses
±trainable_variables
 
 
 
 
µ
´	variables
©layer_metrics
ªlayers
«non_trainable_variables
µregularization_losses
¬metrics
 ­layer_regularization_losses
¶trainable_variables
 

0
1
2

;0
<1
 
 
 
 

-0
.1
 

-0
.1
µ
¿	variables
®layer_metrics
¯layers
°non_trainable_variables
Àregularization_losses
±metrics
 ²layer_regularization_losses
Átrainable_variables
 
 
 

/0
01
=2
>3
 

/0
01
µ
Æ	variables
³layer_metrics
´layers
µnon_trainable_variables
Çregularization_losses
¶metrics
 ·layer_regularization_losses
Ètrainable_variables
 
 
 
 
µ
Ë	variables
¸layer_metrics
¹layers
ºnon_trainable_variables
Ìregularization_losses
»metrics
 ¼layer_regularization_losses
Ítrainable_variables
 

0
1
2

=0
>1
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
 
 
 
 
 
 
 

=0
>1
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
»
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*8
config_proto(&

CPU

GPU2*0J

   E8 *.
f)R'
%__inference_signature_wrapper_6963307
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*;
Tin4
220*
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
 __inference__traced_save_6968077
ñ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betaconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/betaconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/betaconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/betaconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/betaconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/betaconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*:
Tin3
12/*
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
#__inference__traced_restore_6968225æô,
ç
~
)__inference_dense_1_layer_call_fn_6965613

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_69623192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

I__inference_feed_forward_layer_call_and_return_conditional_losses_6965434
dense_input(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense_input#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2 
dropout/dropout/GreaterEqual/yÞ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Mul_1¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::::U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
Ë7

R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6961976

inputs
cnn_block_0_6961883
cnn_block_0_6961885
cnn_block_0_6961887
cnn_block_0_6961889
cnn_block_0_6961891
cnn_block_0_6961893
cnn_block_1_6961896
cnn_block_1_6961898
cnn_block_1_6961900
cnn_block_1_6961902
cnn_block_1_6961904
cnn_block_1_6961906
cnn_block_2_6961909
cnn_block_2_6961911
cnn_block_2_6961913
cnn_block_2_6961915
cnn_block_2_6961917
cnn_block_2_6961919
cnn_block_3_6961922
cnn_block_3_6961924
cnn_block_3_6961926
cnn_block_3_6961928
cnn_block_3_6961930
cnn_block_3_6961932
cnn_block_4_6961935
cnn_block_4_6961937
cnn_block_4_6961939
cnn_block_4_6961941
cnn_block_4_6961943
cnn_block_4_6961945
cnn_block_5_6961948
cnn_block_5_6961950
cnn_block_5_6961952
cnn_block_5_6961954
cnn_block_5_6961956
cnn_block_5_6961958
cnn_block_6_6961961
cnn_block_6_6961963
cnn_block_6_6961965
cnn_block_6_6961967
cnn_block_6_6961969
cnn_block_6_6961971
identity¢#cnn_block_0/StatefulPartitionedCall¢#cnn_block_1/StatefulPartitionedCall¢#cnn_block_2/StatefulPartitionedCall¢#cnn_block_3/StatefulPartitionedCall¢#cnn_block_4/StatefulPartitionedCall¢#cnn_block_5/StatefulPartitionedCall¢#cnn_block_6/StatefulPartitionedCall
#cnn_block_0/StatefulPartitionedCallStatefulPartitionedCallinputscnn_block_0_6961883cnn_block_0_6961885cnn_block_0_6961887cnn_block_0_6961889cnn_block_0_6961891cnn_block_0_6961893*
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
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69595042%
#cnn_block_0/StatefulPartitionedCall¹
#cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_0/StatefulPartitionedCall:output:0cnn_block_1_6961896cnn_block_1_6961898cnn_block_1_6961900cnn_block_1_6961902cnn_block_1_6961904cnn_block_1_6961906*
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
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69598172%
#cnn_block_1/StatefulPartitionedCall¹
#cnn_block_2/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_1/StatefulPartitionedCall:output:0cnn_block_2_6961909cnn_block_2_6961911cnn_block_2_6961913cnn_block_2_6961915cnn_block_2_6961917cnn_block_2_6961919*
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
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69601302%
#cnn_block_2/StatefulPartitionedCall¹
#cnn_block_3/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_2/StatefulPartitionedCall:output:0cnn_block_3_6961922cnn_block_3_6961924cnn_block_3_6961926cnn_block_3_6961928cnn_block_3_6961930cnn_block_3_6961932*
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
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69604432%
#cnn_block_3/StatefulPartitionedCall¹
#cnn_block_4/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_3/StatefulPartitionedCall:output:0cnn_block_4_6961935cnn_block_4_6961937cnn_block_4_6961939cnn_block_4_6961941cnn_block_4_6961943cnn_block_4_6961945*
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
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69607562%
#cnn_block_4/StatefulPartitionedCall¹
#cnn_block_5/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_4/StatefulPartitionedCall:output:0cnn_block_5_6961948cnn_block_5_6961950cnn_block_5_6961952cnn_block_5_6961954cnn_block_5_6961956cnn_block_5_6961958*
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
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69610692%
#cnn_block_5/StatefulPartitionedCallº
#cnn_block_6/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_5/StatefulPartitionedCall:output:0cnn_block_6_6961961cnn_block_6_6961963cnn_block_6_6961965cnn_block_6_6961967cnn_block_6_6961969cnn_block_6_6961971*
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
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69613822%
#cnn_block_6/StatefulPartitionedCall·
(global_average_pooling2d/PartitionedCallPartitionedCall,cnn_block_6/StatefulPartitionedCall:output:0*
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
   E8 *^
fYRW
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_69614402*
(global_average_pooling2d/PartitionedCall
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0$^cnn_block_0/StatefulPartitionedCall$^cnn_block_1/StatefulPartitionedCall$^cnn_block_2/StatefulPartitionedCall$^cnn_block_3/StatefulPartitionedCall$^cnn_block_4/StatefulPartitionedCall$^cnn_block_5/StatefulPartitionedCall$^cnn_block_6/StatefulPartitionedCall*
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

¾
-__inference_cnn_block_1_layer_call_fn_6965957

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
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69598532
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
Ò
`
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6967754

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
 

H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966672
conv2d_6_input+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_6/AssignNewValue¢&batch_normalization_6/AssignNewValue_1±
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpÈ
conv2d_6/Conv2DConv2Dconv2d_6_input&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_6/Conv2D¨
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_6/BiasAdd·
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_6/ReadVariableOp½
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1ê
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ô
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_6/FusedBatchNormV3
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_6/ReluÇ
IdentityIdentityre_lu_6/Relu:activations:0%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_6_input


P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966874

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
ü

H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966758

inputs+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_6/AssignNewValue¢&batch_normalization_6/AssignNewValue_1±
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpÀ
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_6/Conv2D¨
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_6/BiasAdd·
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_6/ReadVariableOp½
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1ê
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ô
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_6/FusedBatchNormV3
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_6/ReluÇ
IdentityIdentityre_lu_6/Relu:activations:0%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¾
-__inference_cnn_block_5_layer_call_fn_6966645

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
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69611052
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
Ò
`
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6961019

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

¾
-__inference_cnn_block_3_layer_call_fn_6966284

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
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69604432
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

¯
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6961273

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

´
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965923

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp¿
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_1/BiasAdd¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_1/Reluv
IdentityIdentityre_lu_1/Relu:activations:0*
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
â
ª
7__inference_batch_normalization_1_layer_call_fn_6967108

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_69597082
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


*__inference_conv2d_5_layer_call_fn_6967621

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
:ÿÿÿÿÿÿÿÿÿ@*$
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
E__inference_conv2d_5_layer_call_and_return_conditional_losses_69609252
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
Ò
`
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6967440

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
í

H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966242

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp¿
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp¬
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_3/BiasAdd¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_3/FusedBatchNormV3
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_3/ReluÆ
IdentityIdentityre_lu_3/Relu:activations:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¾
-__inference_cnn_block_5_layer_call_fn_6966628

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
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69610692
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
¸ø
Í
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6965078
input_15
1cnn_block_0_conv2d_conv2d_readvariableop_resource6
2cnn_block_0_conv2d_biasadd_readvariableop_resource;
7cnn_block_0_batch_normalization_readvariableop_resource=
9cnn_block_0_batch_normalization_readvariableop_1_resourceL
Hcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourceN
Jcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_1_conv2d_1_conv2d_readvariableop_resource8
4cnn_block_1_conv2d_1_biasadd_readvariableop_resource=
9cnn_block_1_batch_normalization_1_readvariableop_resource?
;cnn_block_1_batch_normalization_1_readvariableop_1_resourceN
Jcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_2_conv2d_2_conv2d_readvariableop_resource8
4cnn_block_2_conv2d_2_biasadd_readvariableop_resource=
9cnn_block_2_batch_normalization_2_readvariableop_resource?
;cnn_block_2_batch_normalization_2_readvariableop_1_resourceN
Jcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource8
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource=
9cnn_block_3_batch_normalization_3_readvariableop_resource?
;cnn_block_3_batch_normalization_3_readvariableop_1_resourceN
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource8
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource=
9cnn_block_4_batch_normalization_4_readvariableop_resource?
;cnn_block_4_batch_normalization_4_readvariableop_1_resourceN
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource8
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource=
9cnn_block_5_batch_normalization_5_readvariableop_resource?
;cnn_block_5_batch_normalization_5_readvariableop_1_resourceN
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_6_conv2d_6_conv2d_readvariableop_resource8
4cnn_block_6_conv2d_6_biasadd_readvariableop_resource=
9cnn_block_6_batch_normalization_6_readvariableop_resource?
;cnn_block_6_batch_normalization_6_readvariableop_1_resourceN
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity¢.cnn_block_0/batch_normalization/AssignNewValue¢0cnn_block_0/batch_normalization/AssignNewValue_1¢0cnn_block_1/batch_normalization_1/AssignNewValue¢2cnn_block_1/batch_normalization_1/AssignNewValue_1¢0cnn_block_2/batch_normalization_2/AssignNewValue¢2cnn_block_2/batch_normalization_2/AssignNewValue_1¢0cnn_block_3/batch_normalization_3/AssignNewValue¢2cnn_block_3/batch_normalization_3/AssignNewValue_1¢0cnn_block_4/batch_normalization_4/AssignNewValue¢2cnn_block_4/batch_normalization_4/AssignNewValue_1¢0cnn_block_5/batch_normalization_5/AssignNewValue¢2cnn_block_5/batch_normalization_5/AssignNewValue_1¢0cnn_block_6/batch_normalization_6/AssignNewValue¢2cnn_block_6/batch_normalization_6/AssignNewValue_1Î
(cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOp1cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(cnn_block_0/conv2d/Conv2D/ReadVariableOpÞ
cnn_block_0/conv2d/Conv2DConv2Dinput_10cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_0/conv2d/Conv2DÅ
)cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOp2cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)cnn_block_0/conv2d/BiasAdd/ReadVariableOpÔ
cnn_block_0/conv2d/BiasAddBiasAdd"cnn_block_0/conv2d/Conv2D:output:01cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/conv2d/BiasAddÔ
.cnn_block_0/batch_normalization/ReadVariableOpReadVariableOp7cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype020
.cnn_block_0/batch_normalization/ReadVariableOpÚ
0cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOp9cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype022
0cnn_block_0/batch_normalization/ReadVariableOp_1
?cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp
Acnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Acnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1µ
0cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV3#cnn_block_0/conv2d/BiasAdd:output:06cnn_block_0/batch_normalization/ReadVariableOp:value:08cnn_block_0/batch_normalization/ReadVariableOp_1:value:0Gcnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Icnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<22
0cnn_block_0/batch_normalization/FusedBatchNormV3¿
.cnn_block_0/batch_normalization/AssignNewValueAssignVariableOpHcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource=cnn_block_0/batch_normalization/FusedBatchNormV3:batch_mean:0@^cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp*[
_classQ
OMloc:@cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype020
.cnn_block_0/batch_normalization/AssignNewValueÍ
0cnn_block_0/batch_normalization/AssignNewValue_1AssignVariableOpJcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceAcnn_block_0/batch_normalization/FusedBatchNormV3:batch_variance:0B^cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*]
_classS
QOloc:@cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype022
0cnn_block_0/batch_normalization/AssignNewValue_1¨
cnn_block_0/re_lu/ReluRelu4cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/re_lu/ReluÔ
*cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_1/conv2d_1/Conv2D/ReadVariableOp
cnn_block_1/conv2d_1/Conv2DConv2D$cnn_block_0/re_lu/Relu:activations:02cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_1/conv2d_1/Conv2DË
+cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpÜ
cnn_block_1/conv2d_1/BiasAddBiasAdd$cnn_block_1/conv2d_1/Conv2D:output:03cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/conv2d_1/BiasAddÚ
0cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOp9cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_1/batch_normalization_1/ReadVariableOpà
2cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp;cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_1/batch_normalization_1/ReadVariableOp_1
Acnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Ccnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%cnn_block_1/conv2d_1/BiasAdd:output:08cnn_block_1/batch_normalization_1/ReadVariableOp:value:0:cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0Icnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_1/batch_normalization_1/FusedBatchNormV3Ë
0cnn_block_1/batch_normalization_1/AssignNewValueAssignVariableOpJcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource?cnn_block_1/batch_normalization_1/FusedBatchNormV3:batch_mean:0B^cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_1/batch_normalization_1/AssignNewValueÙ
2cnn_block_1/batch_normalization_1/AssignNewValue_1AssignVariableOpLcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_1/batch_normalization_1/FusedBatchNormV3:batch_variance:0D^cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_1/batch_normalization_1/AssignNewValue_1®
cnn_block_1/re_lu_1/ReluRelu6cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/re_lu_1/ReluÔ
*cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_2/conv2d_2/Conv2D/ReadVariableOp
cnn_block_2/conv2d_2/Conv2DConv2D&cnn_block_1/re_lu_1/Relu:activations:02cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_2/conv2d_2/Conv2DË
+cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpÜ
cnn_block_2/conv2d_2/BiasAddBiasAdd$cnn_block_2/conv2d_2/Conv2D:output:03cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/conv2d_2/BiasAddÚ
0cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOp9cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_2/batch_normalization_2/ReadVariableOpà
2cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp;cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_2/batch_normalization_2/ReadVariableOp_1
Acnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Ccnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%cnn_block_2/conv2d_2/BiasAdd:output:08cnn_block_2/batch_normalization_2/ReadVariableOp:value:0:cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0Icnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_2/batch_normalization_2/FusedBatchNormV3Ë
0cnn_block_2/batch_normalization_2/AssignNewValueAssignVariableOpJcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource?cnn_block_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0B^cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_2/batch_normalization_2/AssignNewValueÙ
2cnn_block_2/batch_normalization_2/AssignNewValue_1AssignVariableOpLcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0D^cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_2/batch_normalization_2/AssignNewValue_1®
cnn_block_2/re_lu_2/ReluRelu6cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/re_lu_2/ReluÔ
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp
cnn_block_3/conv2d_3/Conv2DConv2D&cnn_block_2/re_lu_2/Relu:activations:02cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_3/conv2d_3/Conv2DË
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpÜ
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/conv2d_3/BiasAddÚ
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_3/batch_normalization_3/ReadVariableOpà
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_3/batch_normalization_3/ReadVariableOp_1
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_3/batch_normalization_3/FusedBatchNormV3Ë
0cnn_block_3/batch_normalization_3/AssignNewValueAssignVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_3/batch_normalization_3/AssignNewValueÙ
2cnn_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_3/batch_normalization_3/AssignNewValue_1®
cnn_block_3/re_lu_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/re_lu_3/ReluÔ
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp
cnn_block_4/conv2d_4/Conv2DConv2D&cnn_block_3/re_lu_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_4/conv2d_4/Conv2DË
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpÜ
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/conv2d_4/BiasAddÚ
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_4/batch_normalization_4/ReadVariableOpà
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_4/batch_normalization_4/ReadVariableOp_1
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_4/batch_normalization_4/FusedBatchNormV3Ë
0cnn_block_4/batch_normalization_4/AssignNewValueAssignVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource?cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_4/batch_normalization_4/AssignNewValueÙ
2cnn_block_4/batch_normalization_4/AssignNewValue_1AssignVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0D^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_4/batch_normalization_4/AssignNewValue_1®
cnn_block_4/re_lu_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/re_lu_4/ReluÔ
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp
cnn_block_5/conv2d_5/Conv2DConv2D&cnn_block_4/re_lu_4/Relu:activations:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_5/conv2d_5/Conv2DË
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpÜ
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/conv2d_5/BiasAddÚ
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_5/batch_normalization_5/ReadVariableOpà
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_5/batch_normalization_5/ReadVariableOp_1
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_5/batch_normalization_5/FusedBatchNormV3Ë
0cnn_block_5/batch_normalization_5/AssignNewValueAssignVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_5/batch_normalization_5/AssignNewValueÙ
2cnn_block_5/batch_normalization_5/AssignNewValue_1AssignVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_5/batch_normalization_5/AssignNewValue_1®
cnn_block_5/re_lu_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/re_lu_5/ReluÕ
*cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*cnn_block_6/conv2d_6/Conv2D/ReadVariableOp
cnn_block_6/conv2d_6/Conv2DConv2D&cnn_block_5/re_lu_5/Relu:activations:02cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_6/conv2d_6/Conv2DÌ
+cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpÝ
cnn_block_6/conv2d_6/BiasAddBiasAdd$cnn_block_6/conv2d_6/Conv2D:output:03cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/conv2d_6/BiasAddÛ
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype022
0cnn_block_6/batch_normalization_6/ReadVariableOpá
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2cnn_block_6/batch_normalization_6/ReadVariableOp_1
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1È
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_6/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_6/batch_normalization_6/FusedBatchNormV3Ë
0cnn_block_6/batch_normalization_6/AssignNewValueAssignVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource?cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_6/batch_normalization_6/AssignNewValueÙ
2cnn_block_6/batch_normalization_6/AssignNewValue_1AssignVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0D^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_6/batch_normalization_6/AssignNewValue_1¯
cnn_block_6/re_lu_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/re_lu_6/Relu³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesÛ
global_average_pooling2d/MeanMean&cnn_block_6/re_lu_6/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_average_pooling2d/MeanÏ
IdentityIdentity&global_average_pooling2d/Mean:output:0/^cnn_block_0/batch_normalization/AssignNewValue1^cnn_block_0/batch_normalization/AssignNewValue_11^cnn_block_1/batch_normalization_1/AssignNewValue3^cnn_block_1/batch_normalization_1/AssignNewValue_11^cnn_block_2/batch_normalization_2/AssignNewValue3^cnn_block_2/batch_normalization_2/AssignNewValue_11^cnn_block_3/batch_normalization_3/AssignNewValue3^cnn_block_3/batch_normalization_3/AssignNewValue_11^cnn_block_4/batch_normalization_4/AssignNewValue3^cnn_block_4/batch_normalization_4/AssignNewValue_11^cnn_block_5/batch_normalization_5/AssignNewValue3^cnn_block_5/batch_normalization_5/AssignNewValue_11^cnn_block_6/batch_normalization_6/AssignNewValue3^cnn_block_6/batch_normalization_6/AssignNewValue_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::2`
.cnn_block_0/batch_normalization/AssignNewValue.cnn_block_0/batch_normalization/AssignNewValue2d
0cnn_block_0/batch_normalization/AssignNewValue_10cnn_block_0/batch_normalization/AssignNewValue_12d
0cnn_block_1/batch_normalization_1/AssignNewValue0cnn_block_1/batch_normalization_1/AssignNewValue2h
2cnn_block_1/batch_normalization_1/AssignNewValue_12cnn_block_1/batch_normalization_1/AssignNewValue_12d
0cnn_block_2/batch_normalization_2/AssignNewValue0cnn_block_2/batch_normalization_2/AssignNewValue2h
2cnn_block_2/batch_normalization_2/AssignNewValue_12cnn_block_2/batch_normalization_2/AssignNewValue_12d
0cnn_block_3/batch_normalization_3/AssignNewValue0cnn_block_3/batch_normalization_3/AssignNewValue2h
2cnn_block_3/batch_normalization_3/AssignNewValue_12cnn_block_3/batch_normalization_3/AssignNewValue_12d
0cnn_block_4/batch_normalization_4/AssignNewValue0cnn_block_4/batch_normalization_4/AssignNewValue2h
2cnn_block_4/batch_normalization_4/AssignNewValue_12cnn_block_4/batch_normalization_4/AssignNewValue_12d
0cnn_block_5/batch_normalization_5/AssignNewValue0cnn_block_5/batch_normalization_5/AssignNewValue2h
2cnn_block_5/batch_normalization_5/AssignNewValue_12cnn_block_5/batch_normalization_5/AssignNewValue_12d
0cnn_block_6/batch_normalization_6/AssignNewValue0cnn_block_6/batch_normalization_6/AssignNewValue2h
2cnn_block_6/batch_normalization_6/AssignNewValue_12cnn_block_6/batch_normalization_6/AssignNewValue_1:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

´
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966353

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_4/Conv2D/ReadVariableOp¿
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_4/Conv2D§
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp¬
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_4/BiasAdd¶
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp¼
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1é
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_4/Reluv
IdentityIdentityre_lu_4/Relu:activations:0*
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


R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967188

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
Ö
`
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6967911

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
ª
ª
7__inference_batch_normalization_4_layer_call_fn_6967579

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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_69605562
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

¾
-__inference_cnn_block_6_layer_call_fn_6966817

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
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69614182
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
Z
 
 __inference__traced_save_6968077
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
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
value3B1 B+_temp_e74e24e89cb840ec80000bb56d38fea3/part2	
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
ShardedFilenameõ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*
valueýBú/B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesæ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesã
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/2
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

identity_1Identity_1:output:0*
_input_shapesþ
û: ::::: : : : :  : : : :  : : : : @:@:@:@:@@:@:@:@:@:::::: : : : : : :@:@:@:@:::	@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,	(
&
_output_shapes
:  : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
: :  
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
:@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@:!)

_output_shapes	
::!*

_output_shapes	
::%+!

_output_shapes
:	@: ,

_output_shapes
:@:$- 

_output_shapes

:@
: .

_output_shapes
:
:/

_output_shapes
: 
½
E
)__inference_re_lu_1_layer_call_fn_6967131

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
D__inference_re_lu_1_layer_call_and_return_conditional_losses_69597672
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


R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967345

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

ú
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965726
conv2d_input)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1ª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp¿
conv2d/Conv2DConv2Dconv2d_input$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1á
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2&
$batch_normalization/FusedBatchNormV3÷
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

re_lu/ReluÀ
IdentityIdentityre_lu/Relu:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_1:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
¥
«
C__inference_conv2d_layer_call_and_return_conditional_losses_6959360

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

V
:__inference_global_average_pooling2d_layer_call_fn_6961446

inputs
identityç
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
   E8 *^
fYRW
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_69614402
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
ª
ª
B__inference_dense_layer_call_and_return_conditional_losses_6965558

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
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
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
­
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6959986

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
§
­
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6967141

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
Ñ

R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967502

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
£â
ó'
F__inference_class_cnn_layer_call_and_return_conditional_losses_6963497
input_1K
Gfeature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resourceL
Hfeature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resourceQ
Mfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resourceb
^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourced
`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource5
1feed_forward_dense_matmul_readvariableop_resource6
2feed_forward_dense_biasadd_readvariableop_resource7
3feed_forward_dense_1_matmul_readvariableop_resource8
4feed_forward_dense_1_biasadd_readvariableop_resource
identity¢Dfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue¢Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1[
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
truediv
>feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOpGfeature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02@
>feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp¤
/feature_extractor_cnn/cnn_block_0/conv2d/Conv2DConv2Dtruediv:z:0Ffeature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
21
/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D
?feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOpHfeature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp¬
0feature_extractor_cnn/cnn_block_0/conv2d/BiasAddBiasAdd8feature_extractor_cnn/cnn_block_0/conv2d/Conv2D:output:0Gfeature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOpReadVariableOpMfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02F
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOpOfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1É
Ufeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02W
Ufeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpÏ
Wfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02Y
Wfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ï
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV39feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd:output:0Lfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp:value:0Nfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1:value:0]feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0_feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3Ã
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValueAssignVariableOp^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourceSfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:batch_mean:0V^feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp*q
_classg
ecloc:@feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02F
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValueÑ
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1AssignVariableOp`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceWfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:batch_variance:0X^feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*s
_classi
geloc:@feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1ê
,feature_extractor_cnn/cnn_block_0/re_lu/ReluReluJfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,feature_extractor_cnn/cnn_block_0/re_lu/Relu
@feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02B
@feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpÙ
1feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2DConv2D:feature_extractor_cnn/cnn_block_0/re_lu/Relu:activations:0Hfeature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D
Afeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_1/re_lu_1/ReluReluLfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_1/re_lu_1/Relu
@feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02B
@feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2DConv2D<feature_extractor_cnn/cnn_block_1/re_lu_1/Relu:activations:0Hfeature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D
Afeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_2/re_lu_2/ReluReluLfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_2/re_lu_2/Relu
@feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02B
@feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2DConv2D<feature_extractor_cnn/cnn_block_2/re_lu_2/Relu:activations:0Hfeature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D
Afeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_3/re_lu_3/ReluReluLfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_3/re_lu_3/Relu
@feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2DConv2D<feature_extractor_cnn/cnn_block_3/re_lu_3/Relu:activations:0Hfeature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D
Afeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Afeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02H
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Y
Wfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02[
Yfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_4/re_lu_4/ReluReluLfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.feature_extractor_cnn/cnn_block_4/re_lu_4/Relu
@feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02B
@feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2DConv2D<feature_extractor_cnn/cnn_block_4/re_lu_4/Relu:activations:0Hfeature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D
Afeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Afeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02H
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Y
Wfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02[
Yfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_5/re_lu_5/ReluReluLfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.feature_extractor_cnn/cnn_block_5/re_lu_5/Relu
@feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02B
@feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpÜ
1feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2DConv2D<feature_extractor_cnn/cnn_block_5/re_lu_5/Relu:activations:0Hfeature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D
Afeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02C
Afeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpµ
2feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02H
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp£
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1Ð
Wfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02Y
Wfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpÖ
Yfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02[
Yfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1â
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1ñ
.feature_extractor_cnn/cnn_block_6/re_lu_6/ReluReluLfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.feature_extractor_cnn/cnn_block_6/re_lu_6/Reluß
Efeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2G
Efeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices³
3feature_extractor_cnn/global_average_pooling2d/MeanMean<feature_extractor_cnn/cnn_block_6/re_lu_6/Relu:activations:0Nfeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3feature_extractor_cnn/global_average_pooling2d/MeanÇ
(feed_forward/dense/MatMul/ReadVariableOpReadVariableOp1feed_forward_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02*
(feed_forward/dense/MatMul/ReadVariableOpâ
feed_forward/dense/MatMulMatMul<feature_extractor_cnn/global_average_pooling2d/Mean:output:00feed_forward/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/MatMulÅ
)feed_forward/dense/BiasAdd/ReadVariableOpReadVariableOp2feed_forward_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)feed_forward/dense/BiasAdd/ReadVariableOpÍ
feed_forward/dense/BiasAddBiasAdd#feed_forward/dense/MatMul:product:01feed_forward/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/BiasAdd
feed_forward/dense/ReluRelu#feed_forward/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/Relu
"feed_forward/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"feed_forward/dropout/dropout/ConstÑ
 feed_forward/dropout/dropout/MulMul%feed_forward/dense/Relu:activations:0+feed_forward/dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 feed_forward/dropout/dropout/Mul
"feed_forward/dropout/dropout/ShapeShape%feed_forward/dense/Relu:activations:0*
T0*
_output_shapes
:2$
"feed_forward/dropout/dropout/Shapeó
9feed_forward/dropout/dropout/random_uniform/RandomUniformRandomUniform+feed_forward/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02;
9feed_forward/dropout/dropout/random_uniform/RandomUniform
+feed_forward/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2-
+feed_forward/dropout/dropout/GreaterEqual/y
)feed_forward/dropout/dropout/GreaterEqualGreaterEqualBfeed_forward/dropout/dropout/random_uniform/RandomUniform:output:04feed_forward/dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)feed_forward/dropout/dropout/GreaterEqual¾
!feed_forward/dropout/dropout/CastCast-feed_forward/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!feed_forward/dropout/dropout/CastÎ
"feed_forward/dropout/dropout/Mul_1Mul$feed_forward/dropout/dropout/Mul:z:0%feed_forward/dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"feed_forward/dropout/dropout/Mul_1Ì
*feed_forward/dense_1/MatMul/ReadVariableOpReadVariableOp3feed_forward_dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02,
*feed_forward/dense_1/MatMul/ReadVariableOpÒ
feed_forward/dense_1/MatMulMatMul&feed_forward/dropout/dropout/Mul_1:z:02feed_forward/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
feed_forward/dense_1/MatMulË
+feed_forward/dense_1/BiasAdd/ReadVariableOpReadVariableOp4feed_forward_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+feed_forward/dense_1/BiasAdd/ReadVariableOpÕ
feed_forward/dense_1/BiasAddBiasAdd%feed_forward/dense_1/MatMul:product:03feed_forward/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
feed_forward/dense_1/BiasAdd	
IdentityIdentity%feed_forward/dense_1/BiasAdd:output:0E^feature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValueG^feature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1G^feature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValueI^feature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1G^feature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValueI^feature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1G^feature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValueI^feature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1G^feature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValueI^feature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1G^feature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValueI^feature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1G^feature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValueI^feature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::2
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValueDfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue2
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValueFfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValueFfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValueFfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValueFfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValueFfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValueFfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ô­
ø
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6964733

inputs5
1cnn_block_0_conv2d_conv2d_readvariableop_resource6
2cnn_block_0_conv2d_biasadd_readvariableop_resource;
7cnn_block_0_batch_normalization_readvariableop_resource=
9cnn_block_0_batch_normalization_readvariableop_1_resourceL
Hcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourceN
Jcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_1_conv2d_1_conv2d_readvariableop_resource8
4cnn_block_1_conv2d_1_biasadd_readvariableop_resource=
9cnn_block_1_batch_normalization_1_readvariableop_resource?
;cnn_block_1_batch_normalization_1_readvariableop_1_resourceN
Jcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_2_conv2d_2_conv2d_readvariableop_resource8
4cnn_block_2_conv2d_2_biasadd_readvariableop_resource=
9cnn_block_2_batch_normalization_2_readvariableop_resource?
;cnn_block_2_batch_normalization_2_readvariableop_1_resourceN
Jcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource8
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource=
9cnn_block_3_batch_normalization_3_readvariableop_resource?
;cnn_block_3_batch_normalization_3_readvariableop_1_resourceN
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource8
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource=
9cnn_block_4_batch_normalization_4_readvariableop_resource?
;cnn_block_4_batch_normalization_4_readvariableop_1_resourceN
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource8
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource=
9cnn_block_5_batch_normalization_5_readvariableop_resource?
;cnn_block_5_batch_normalization_5_readvariableop_1_resourceN
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_6_conv2d_6_conv2d_readvariableop_resource8
4cnn_block_6_conv2d_6_biasadd_readvariableop_resource=
9cnn_block_6_batch_normalization_6_readvariableop_resource?
;cnn_block_6_batch_normalization_6_readvariableop_1_resourceN
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identityÎ
(cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOp1cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(cnn_block_0/conv2d/Conv2D/ReadVariableOpÝ
cnn_block_0/conv2d/Conv2DConv2Dinputs0cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_0/conv2d/Conv2DÅ
)cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOp2cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)cnn_block_0/conv2d/BiasAdd/ReadVariableOpÔ
cnn_block_0/conv2d/BiasAddBiasAdd"cnn_block_0/conv2d/Conv2D:output:01cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/conv2d/BiasAddÔ
.cnn_block_0/batch_normalization/ReadVariableOpReadVariableOp7cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype020
.cnn_block_0/batch_normalization/ReadVariableOpÚ
0cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOp9cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype022
0cnn_block_0/batch_normalization/ReadVariableOp_1
?cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp
Acnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Acnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1§
0cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV3#cnn_block_0/conv2d/BiasAdd:output:06cnn_block_0/batch_normalization/ReadVariableOp:value:08cnn_block_0/batch_normalization/ReadVariableOp_1:value:0Gcnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Icnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 22
0cnn_block_0/batch_normalization/FusedBatchNormV3¨
cnn_block_0/re_lu/ReluRelu4cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/re_lu/ReluÔ
*cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_1/conv2d_1/Conv2D/ReadVariableOp
cnn_block_1/conv2d_1/Conv2DConv2D$cnn_block_0/re_lu/Relu:activations:02cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_1/conv2d_1/Conv2DË
+cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpÜ
cnn_block_1/conv2d_1/BiasAddBiasAdd$cnn_block_1/conv2d_1/Conv2D:output:03cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/conv2d_1/BiasAddÚ
0cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOp9cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_1/batch_normalization_1/ReadVariableOpà
2cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp;cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_1/batch_normalization_1/ReadVariableOp_1
Acnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Ccnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%cnn_block_1/conv2d_1/BiasAdd:output:08cnn_block_1/batch_normalization_1/ReadVariableOp:value:0:cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0Icnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_1/batch_normalization_1/FusedBatchNormV3®
cnn_block_1/re_lu_1/ReluRelu6cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/re_lu_1/ReluÔ
*cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_2/conv2d_2/Conv2D/ReadVariableOp
cnn_block_2/conv2d_2/Conv2DConv2D&cnn_block_1/re_lu_1/Relu:activations:02cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_2/conv2d_2/Conv2DË
+cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpÜ
cnn_block_2/conv2d_2/BiasAddBiasAdd$cnn_block_2/conv2d_2/Conv2D:output:03cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/conv2d_2/BiasAddÚ
0cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOp9cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_2/batch_normalization_2/ReadVariableOpà
2cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp;cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_2/batch_normalization_2/ReadVariableOp_1
Acnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Ccnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%cnn_block_2/conv2d_2/BiasAdd:output:08cnn_block_2/batch_normalization_2/ReadVariableOp:value:0:cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0Icnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_2/batch_normalization_2/FusedBatchNormV3®
cnn_block_2/re_lu_2/ReluRelu6cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/re_lu_2/ReluÔ
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp
cnn_block_3/conv2d_3/Conv2DConv2D&cnn_block_2/re_lu_2/Relu:activations:02cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_3/conv2d_3/Conv2DË
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpÜ
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/conv2d_3/BiasAddÚ
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_3/batch_normalization_3/ReadVariableOpà
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_3/batch_normalization_3/ReadVariableOp_1
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_3/batch_normalization_3/FusedBatchNormV3®
cnn_block_3/re_lu_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/re_lu_3/ReluÔ
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp
cnn_block_4/conv2d_4/Conv2DConv2D&cnn_block_3/re_lu_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_4/conv2d_4/Conv2DË
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpÜ
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/conv2d_4/BiasAddÚ
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_4/batch_normalization_4/ReadVariableOpà
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_4/batch_normalization_4/ReadVariableOp_1
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 24
2cnn_block_4/batch_normalization_4/FusedBatchNormV3®
cnn_block_4/re_lu_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/re_lu_4/ReluÔ
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp
cnn_block_5/conv2d_5/Conv2DConv2D&cnn_block_4/re_lu_4/Relu:activations:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_5/conv2d_5/Conv2DË
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpÜ
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/conv2d_5/BiasAddÚ
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_5/batch_normalization_5/ReadVariableOpà
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_5/batch_normalization_5/ReadVariableOp_1
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 24
2cnn_block_5/batch_normalization_5/FusedBatchNormV3®
cnn_block_5/re_lu_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/re_lu_5/ReluÕ
*cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*cnn_block_6/conv2d_6/Conv2D/ReadVariableOp
cnn_block_6/conv2d_6/Conv2DConv2D&cnn_block_5/re_lu_5/Relu:activations:02cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_6/conv2d_6/Conv2DÌ
+cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpÝ
cnn_block_6/conv2d_6/BiasAddBiasAdd$cnn_block_6/conv2d_6/Conv2D:output:03cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/conv2d_6/BiasAddÛ
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype022
0cnn_block_6/batch_normalization_6/ReadVariableOpá
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2cnn_block_6/batch_normalization_6/ReadVariableOp_1
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1º
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_6/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 24
2cnn_block_6/batch_normalization_6/FusedBatchNormV3¯
cnn_block_6/re_lu_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/re_lu_6/Relu³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesÛ
global_average_pooling2d/MeanMean&cnn_block_6/re_lu_6/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_average_pooling2d/Mean{
IdentityIdentity&global_average_pooling2d/Mean:output:0*
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
·
·
+__inference_class_cnn_layer_call_fn_6964316

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

unknown_41

unknown_42

unknown_43

unknown_44
identity¢StatefulPartitionedCallÒ
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
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 !"%&'(+,-.*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_class_cnn_layer_call_and_return_conditional_losses_69629162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
b
D__inference_dropout_layer_call_and_return_conditional_losses_6962296

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967170

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
§
­
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6966984

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
º
¸
+__inference_class_cnn_layer_call_fn_6963763
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

unknown_41

unknown_42

unknown_43

unknown_44
identity¢StatefulPartitionedCallÓ
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
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*B
_read_only_resource_inputs$
" 	
 !"%&'(+,-.*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_class_cnn_layer_call_and_return_conditional_losses_69629162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ñ

R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967723

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


*__inference_conv2d_4_layer_call_fn_6967464

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
:ÿÿÿÿÿÿÿÿÿ@*$
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
E__inference_conv2d_4_layer_call_and_return_conditional_losses_69606122
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
Ì

7__inference_feature_extractor_cnn_layer_call_fn_6965320
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
identity¢StatefulPartitionedCall¨
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
   E8 *[
fVRT
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_69619762
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
¬
ª
7__inference_batch_normalization_1_layer_call_fn_6967057

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_69596482
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
¬
­
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6967769

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


*__inference_conv2d_3_layer_call_fn_6967307

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
:ÿÿÿÿÿÿÿÿÿ *$
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
E__inference_conv2d_3_layer_call_and_return_conditional_losses_69602992
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
¨
¨
5__inference_batch_normalization_layer_call_fn_6966900

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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_69593352
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
Ê
¯
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967641

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
»
q
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_6961440

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
¬
­
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6961238

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
Ê
¯
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6960869

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
¯ô
ë
F__inference_class_cnn_layer_call_and_return_conditional_losses_6963666
input_1K
Gfeature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resourceL
Hfeature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resourceQ
Mfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resourceb
^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourced
`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource5
1feed_forward_dense_matmul_readvariableop_resource6
2feed_forward_dense_biasadd_readvariableop_resource7
3feed_forward_dense_1_matmul_readvariableop_resource8
4feed_forward_dense_1_biasadd_readvariableop_resource
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
truediv
>feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOpGfeature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02@
>feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp¤
/feature_extractor_cnn/cnn_block_0/conv2d/Conv2DConv2Dtruediv:z:0Ffeature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
21
/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D
?feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOpHfeature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp¬
0feature_extractor_cnn/cnn_block_0/conv2d/BiasAddBiasAdd8feature_extractor_cnn/cnn_block_0/conv2d/Conv2D:output:0Gfeature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOpReadVariableOpMfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02F
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOpOfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1É
Ufeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02W
Ufeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpÏ
Wfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02Y
Wfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Á
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV39feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd:output:0Lfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp:value:0Nfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1:value:0]feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0_feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3ê
,feature_extractor_cnn/cnn_block_0/re_lu/ReluReluJfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,feature_extractor_cnn/cnn_block_0/re_lu/Relu
@feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02B
@feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpÙ
1feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2DConv2D:feature_extractor_cnn/cnn_block_0/re_lu/Relu:activations:0Hfeature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D
Afeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_1/re_lu_1/ReluReluLfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_1/re_lu_1/Relu
@feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02B
@feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2DConv2D<feature_extractor_cnn/cnn_block_1/re_lu_1/Relu:activations:0Hfeature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D
Afeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_2/re_lu_2/ReluReluLfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_2/re_lu_2/Relu
@feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02B
@feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2DConv2D<feature_extractor_cnn/cnn_block_2/re_lu_2/Relu:activations:0Hfeature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D
Afeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_3/re_lu_3/ReluReluLfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_3/re_lu_3/Relu
@feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2DConv2D<feature_extractor_cnn/cnn_block_3/re_lu_3/Relu:activations:0Hfeature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D
Afeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Afeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02H
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Y
Wfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02[
Yfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_4/re_lu_4/ReluReluLfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.feature_extractor_cnn/cnn_block_4/re_lu_4/Relu
@feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02B
@feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2DConv2D<feature_extractor_cnn/cnn_block_4/re_lu_4/Relu:activations:0Hfeature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D
Afeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Afeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02H
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Y
Wfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02[
Yfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_5/re_lu_5/ReluReluLfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.feature_extractor_cnn/cnn_block_5/re_lu_5/Relu
@feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02B
@feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpÜ
1feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2DConv2D<feature_extractor_cnn/cnn_block_5/re_lu_5/Relu:activations:0Hfeature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D
Afeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02C
Afeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpµ
2feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02H
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp£
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1Ð
Wfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02Y
Wfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpÖ
Yfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02[
Yfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ô
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3ñ
.feature_extractor_cnn/cnn_block_6/re_lu_6/ReluReluLfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.feature_extractor_cnn/cnn_block_6/re_lu_6/Reluß
Efeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2G
Efeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices³
3feature_extractor_cnn/global_average_pooling2d/MeanMean<feature_extractor_cnn/cnn_block_6/re_lu_6/Relu:activations:0Nfeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3feature_extractor_cnn/global_average_pooling2d/MeanÇ
(feed_forward/dense/MatMul/ReadVariableOpReadVariableOp1feed_forward_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02*
(feed_forward/dense/MatMul/ReadVariableOpâ
feed_forward/dense/MatMulMatMul<feature_extractor_cnn/global_average_pooling2d/Mean:output:00feed_forward/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/MatMulÅ
)feed_forward/dense/BiasAdd/ReadVariableOpReadVariableOp2feed_forward_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)feed_forward/dense/BiasAdd/ReadVariableOpÍ
feed_forward/dense/BiasAddBiasAdd#feed_forward/dense/MatMul:product:01feed_forward/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/BiasAdd
feed_forward/dense/ReluRelu#feed_forward/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/Relu£
feed_forward/dropout/IdentityIdentity%feed_forward/dense/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dropout/IdentityÌ
*feed_forward/dense_1/MatMul/ReadVariableOpReadVariableOp3feed_forward_dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02,
*feed_forward/dense_1/MatMul/ReadVariableOpÒ
feed_forward/dense_1/MatMulMatMul&feed_forward/dropout/Identity:output:02feed_forward/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
feed_forward/dense_1/MatMulË
+feed_forward/dense_1/BiasAdd/ReadVariableOpReadVariableOp4feed_forward_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+feed_forward/dense_1/BiasAdd/ReadVariableOpÕ
feed_forward/dense_1/BiasAddBiasAdd%feed_forward/dense_1/MatMul:product:03feed_forward/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
feed_forward/dense_1/BiasAddy
IdentityIdentity%feed_forward/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6959648

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

¯
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967484

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

¼
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966439
conv2d_4_input+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_4/Conv2D/ReadVariableOpÇ
conv2d_4/Conv2DConv2Dconv2d_4_input&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_4/Conv2D§
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp¬
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_4/BiasAdd¶
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp¼
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1é
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_4/Reluv
IdentityIdentityre_lu_4/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :::::::_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_4_input
Ê
¯
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967548

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

¾
-__inference_cnn_block_1_layer_call_fn_6965940

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
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69598172
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
 

H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966500
conv2d_5_input+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_5/AssignNewValue¢&batch_normalization_5/AssignNewValue_1°
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpÇ
conv2d_5/Conv2DConv2Dconv2d_5_input&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_5/Conv2D§
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp¬
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_5/BiasAdd¶
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp¼
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1é
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_5/FusedBatchNormV3
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_5/ReluÆ
IdentityIdentityre_lu_5/Relu:activations:0%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_5_input
¦
Æ
-__inference_cnn_block_5_layer_call_fn_6966542
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69610692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_5_input
Ù7

R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6962161

inputs
cnn_block_0_6962068
cnn_block_0_6962070
cnn_block_0_6962072
cnn_block_0_6962074
cnn_block_0_6962076
cnn_block_0_6962078
cnn_block_1_6962081
cnn_block_1_6962083
cnn_block_1_6962085
cnn_block_1_6962087
cnn_block_1_6962089
cnn_block_1_6962091
cnn_block_2_6962094
cnn_block_2_6962096
cnn_block_2_6962098
cnn_block_2_6962100
cnn_block_2_6962102
cnn_block_2_6962104
cnn_block_3_6962107
cnn_block_3_6962109
cnn_block_3_6962111
cnn_block_3_6962113
cnn_block_3_6962115
cnn_block_3_6962117
cnn_block_4_6962120
cnn_block_4_6962122
cnn_block_4_6962124
cnn_block_4_6962126
cnn_block_4_6962128
cnn_block_4_6962130
cnn_block_5_6962133
cnn_block_5_6962135
cnn_block_5_6962137
cnn_block_5_6962139
cnn_block_5_6962141
cnn_block_5_6962143
cnn_block_6_6962146
cnn_block_6_6962148
cnn_block_6_6962150
cnn_block_6_6962152
cnn_block_6_6962154
cnn_block_6_6962156
identity¢#cnn_block_0/StatefulPartitionedCall¢#cnn_block_1/StatefulPartitionedCall¢#cnn_block_2/StatefulPartitionedCall¢#cnn_block_3/StatefulPartitionedCall¢#cnn_block_4/StatefulPartitionedCall¢#cnn_block_5/StatefulPartitionedCall¢#cnn_block_6/StatefulPartitionedCall
#cnn_block_0/StatefulPartitionedCallStatefulPartitionedCallinputscnn_block_0_6962068cnn_block_0_6962070cnn_block_0_6962072cnn_block_0_6962074cnn_block_0_6962076cnn_block_0_6962078*
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
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69595402%
#cnn_block_0/StatefulPartitionedCall»
#cnn_block_1/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_0/StatefulPartitionedCall:output:0cnn_block_1_6962081cnn_block_1_6962083cnn_block_1_6962085cnn_block_1_6962087cnn_block_1_6962089cnn_block_1_6962091*
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
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69598532%
#cnn_block_1/StatefulPartitionedCall»
#cnn_block_2/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_1/StatefulPartitionedCall:output:0cnn_block_2_6962094cnn_block_2_6962096cnn_block_2_6962098cnn_block_2_6962100cnn_block_2_6962102cnn_block_2_6962104*
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
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69601662%
#cnn_block_2/StatefulPartitionedCall»
#cnn_block_3/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_2/StatefulPartitionedCall:output:0cnn_block_3_6962107cnn_block_3_6962109cnn_block_3_6962111cnn_block_3_6962113cnn_block_3_6962115cnn_block_3_6962117*
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
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69604792%
#cnn_block_3/StatefulPartitionedCall»
#cnn_block_4/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_3/StatefulPartitionedCall:output:0cnn_block_4_6962120cnn_block_4_6962122cnn_block_4_6962124cnn_block_4_6962126cnn_block_4_6962128cnn_block_4_6962130*
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
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69607922%
#cnn_block_4/StatefulPartitionedCall»
#cnn_block_5/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_4/StatefulPartitionedCall:output:0cnn_block_5_6962133cnn_block_5_6962135cnn_block_5_6962137cnn_block_5_6962139cnn_block_5_6962141cnn_block_5_6962143*
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
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69611052%
#cnn_block_5/StatefulPartitionedCall¼
#cnn_block_6/StatefulPartitionedCallStatefulPartitionedCall,cnn_block_5/StatefulPartitionedCall:output:0cnn_block_6_6962146cnn_block_6_6962148cnn_block_6_6962150cnn_block_6_6962152cnn_block_6_6962154cnn_block_6_6962156*
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
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69614182%
#cnn_block_6/StatefulPartitionedCall·
(global_average_pooling2d/PartitionedCallPartitionedCall,cnn_block_6/StatefulPartitionedCall:output:0*
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
   E8 *^
fYRW
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_69614402*
(global_average_pooling2d/PartitionedCall
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0$^cnn_block_0/StatefulPartitionedCall$^cnn_block_1/StatefulPartitionedCall$^cnn_block_2/StatefulPartitionedCall$^cnn_block_3/StatefulPartitionedCall$^cnn_block_4/StatefulPartitionedCall$^cnn_block_5/StatefulPartitionedCall$^cnn_block_6/StatefulPartitionedCall*
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
Ò
`
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6960393

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

â
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6959504

inputs
conv2d_6959488
conv2d_6959490
batch_normalization_6959493
batch_normalization_6959495
batch_normalization_6959497
batch_normalization_6959499
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall 
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6959488conv2d_6959490*
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
   E8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_69593602 
conv2d/StatefulPartitionedCall¾
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_6959493batch_normalization_6959495batch_normalization_6959497batch_normalization_6959499*
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
   E8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_69593952-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
   E8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_69594542
re_lu/PartitionedCallÉ
IdentityIdentityre_lu/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
-__inference_cnn_block_2_layer_call_fn_6966112

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
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69601302
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
Ã
ò
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6960130

inputs
conv2d_2_6960114
conv2d_2_6960116!
batch_normalization_2_6960119!
batch_normalization_2_6960121!
batch_normalization_2_6960123!
batch_normalization_2_6960125
identity¢-batch_normalization_2/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCallª
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_6960114conv2d_2_6960116*
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
E__inference_conv2d_2_layer_call_and_return_conditional_losses_69599862"
 conv2d_2/StatefulPartitionedCallÎ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_6960119batch_normalization_2_6960121batch_normalization_2_6960123batch_normalization_2_6960125*
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_69600212/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_2_layer_call_and_return_conditional_losses_69600802
re_lu_2/PartitionedCallÏ
IdentityIdentity re_lu_2/PartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§
­
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6960612

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
´
¡
.__inference_feed_forward_layer_call_fn_6965534

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *R
fMRK
I__inference_feed_forward_layer_call_and_return_conditional_losses_69623692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
ª
7__inference_batch_normalization_3_layer_call_fn_6967435

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
:ÿÿÿÿÿÿÿÿÿ *&
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_69603522
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
Ê
¯
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6960556

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
æ
ª
7__inference_batch_normalization_6_layer_call_fn_6967893

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
   E8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69612732
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
Â'
â
F__inference_class_cnn_layer_call_and_return_conditional_losses_6962916

inputs!
feature_extractor_cnn_6962821!
feature_extractor_cnn_6962823!
feature_extractor_cnn_6962825!
feature_extractor_cnn_6962827!
feature_extractor_cnn_6962829!
feature_extractor_cnn_6962831!
feature_extractor_cnn_6962833!
feature_extractor_cnn_6962835!
feature_extractor_cnn_6962837!
feature_extractor_cnn_6962839!
feature_extractor_cnn_6962841!
feature_extractor_cnn_6962843!
feature_extractor_cnn_6962845!
feature_extractor_cnn_6962847!
feature_extractor_cnn_6962849!
feature_extractor_cnn_6962851!
feature_extractor_cnn_6962853!
feature_extractor_cnn_6962855!
feature_extractor_cnn_6962857!
feature_extractor_cnn_6962859!
feature_extractor_cnn_6962861!
feature_extractor_cnn_6962863!
feature_extractor_cnn_6962865!
feature_extractor_cnn_6962867!
feature_extractor_cnn_6962869!
feature_extractor_cnn_6962871!
feature_extractor_cnn_6962873!
feature_extractor_cnn_6962875!
feature_extractor_cnn_6962877!
feature_extractor_cnn_6962879!
feature_extractor_cnn_6962881!
feature_extractor_cnn_6962883!
feature_extractor_cnn_6962885!
feature_extractor_cnn_6962887!
feature_extractor_cnn_6962889!
feature_extractor_cnn_6962891!
feature_extractor_cnn_6962893!
feature_extractor_cnn_6962895!
feature_extractor_cnn_6962897!
feature_extractor_cnn_6962899!
feature_extractor_cnn_6962901!
feature_extractor_cnn_6962903
feed_forward_6962906
feed_forward_6962908
feed_forward_6962910
feed_forward_6962912
identity¢-feature_extractor_cnn/StatefulPartitionedCall¢$feed_forward/StatefulPartitionedCall[
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
truediv
-feature_extractor_cnn/StatefulPartitionedCallStatefulPartitionedCalltruediv:z:0feature_extractor_cnn_6962821feature_extractor_cnn_6962823feature_extractor_cnn_6962825feature_extractor_cnn_6962827feature_extractor_cnn_6962829feature_extractor_cnn_6962831feature_extractor_cnn_6962833feature_extractor_cnn_6962835feature_extractor_cnn_6962837feature_extractor_cnn_6962839feature_extractor_cnn_6962841feature_extractor_cnn_6962843feature_extractor_cnn_6962845feature_extractor_cnn_6962847feature_extractor_cnn_6962849feature_extractor_cnn_6962851feature_extractor_cnn_6962853feature_extractor_cnn_6962855feature_extractor_cnn_6962857feature_extractor_cnn_6962859feature_extractor_cnn_6962861feature_extractor_cnn_6962863feature_extractor_cnn_6962865feature_extractor_cnn_6962867feature_extractor_cnn_6962869feature_extractor_cnn_6962871feature_extractor_cnn_6962873feature_extractor_cnn_6962875feature_extractor_cnn_6962877feature_extractor_cnn_6962879feature_extractor_cnn_6962881feature_extractor_cnn_6962883feature_extractor_cnn_6962885feature_extractor_cnn_6962887feature_extractor_cnn_6962889feature_extractor_cnn_6962891feature_extractor_cnn_6962893feature_extractor_cnn_6962895feature_extractor_cnn_6962897feature_extractor_cnn_6962899feature_extractor_cnn_6962901feature_extractor_cnn_6962903*6
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
   E8 *[
fVRT
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_69619762/
-feature_extractor_cnn/StatefulPartitionedCall
$feed_forward/StatefulPartitionedCallStatefulPartitionedCall6feature_extractor_cnn/StatefulPartitionedCall:output:0feed_forward_6962906feed_forward_6962908feed_forward_6962910feed_forward_6962912*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *R
fMRK
I__inference_feed_forward_layer_call_and_return_conditional_losses_69623692&
$feed_forward/StatefulPartitionedCallØ
IdentityIdentity-feed_forward/StatefulPartitionedCall:output:0.^feature_extractor_cnn/StatefulPartitionedCall%^feed_forward/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::2^
-feature_extractor_cnn/StatefulPartitionedCall-feature_extractor_cnn/StatefulPartitionedCall2L
$feed_forward/StatefulPartitionedCall$feed_forward/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
Æ
-__inference_cnn_block_1_layer_call_fn_6965854
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69598172
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
_user_specified_nameconv2d_1_input
®
ª
7__inference_batch_normalization_6_layer_call_fn_6967829

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
   E8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69611822
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
Ú

7__inference_feature_extractor_cnn_layer_call_fn_6965409
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
identity¢StatefulPartitionedCall¶
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
   E8 *[
fVRT
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_69621612
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

¯
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6960960

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
È
­
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6959304

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
Æ
-__inference_cnn_block_1_layer_call_fn_6965871
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_69598532
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
_user_specified_nameconv2d_1_input


I__inference_feed_forward_layer_call_and_return_conditional_losses_6965503

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2 
dropout/dropout/GreaterEqual/yÞ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Mul_1¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 â
ò'
F__inference_class_cnn_layer_call_and_return_conditional_losses_6964050

inputsK
Gfeature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resourceL
Hfeature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resourceQ
Mfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resourceb
^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourced
`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource5
1feed_forward_dense_matmul_readvariableop_resource6
2feed_forward_dense_biasadd_readvariableop_resource7
3feed_forward_dense_1_matmul_readvariableop_resource8
4feed_forward_dense_1_biasadd_readvariableop_resource
identity¢Dfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue¢Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1¢Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue¢Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1[
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
truediv
>feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOpGfeature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02@
>feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp¤
/feature_extractor_cnn/cnn_block_0/conv2d/Conv2DConv2Dtruediv:z:0Ffeature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
21
/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D
?feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOpHfeature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp¬
0feature_extractor_cnn/cnn_block_0/conv2d/BiasAddBiasAdd8feature_extractor_cnn/cnn_block_0/conv2d/Conv2D:output:0Gfeature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOpReadVariableOpMfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02F
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOpOfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1É
Ufeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02W
Ufeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpÏ
Wfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02Y
Wfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ï
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV39feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd:output:0Lfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp:value:0Nfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1:value:0]feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0_feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3Ã
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValueAssignVariableOp^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourceSfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:batch_mean:0V^feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp*q
_classg
ecloc:@feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02F
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValueÑ
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1AssignVariableOp`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceWfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:batch_variance:0X^feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*s
_classi
geloc:@feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1ê
,feature_extractor_cnn/cnn_block_0/re_lu/ReluReluJfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,feature_extractor_cnn/cnn_block_0/re_lu/Relu
@feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02B
@feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpÙ
1feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2DConv2D:feature_extractor_cnn/cnn_block_0/re_lu/Relu:activations:0Hfeature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D
Afeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_1/re_lu_1/ReluReluLfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_1/re_lu_1/Relu
@feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02B
@feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2DConv2D<feature_extractor_cnn/cnn_block_1/re_lu_1/Relu:activations:0Hfeature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D
Afeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_2/re_lu_2/ReluReluLfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_2/re_lu_2/Relu
@feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02B
@feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2DConv2D<feature_extractor_cnn/cnn_block_2/re_lu_2/Relu:activations:0Hfeature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D
Afeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_3/re_lu_3/ReluReluLfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_3/re_lu_3/Relu
@feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2DConv2D<feature_extractor_cnn/cnn_block_3/re_lu_3/Relu:activations:0Hfeature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D
Afeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Afeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02H
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Y
Wfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02[
Yfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_4/re_lu_4/ReluReluLfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.feature_extractor_cnn/cnn_block_4/re_lu_4/Relu
@feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02B
@feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2DConv2D<feature_extractor_cnn/cnn_block_4/re_lu_4/Relu:activations:0Hfeature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D
Afeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Afeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02H
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Y
Wfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02[
Yfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ý
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1ð
.feature_extractor_cnn/cnn_block_5/re_lu_5/ReluReluLfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.feature_extractor_cnn/cnn_block_5/re_lu_5/Relu
@feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02B
@feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpÜ
1feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2DConv2D<feature_extractor_cnn/cnn_block_5/re_lu_5/Relu:activations:0Hfeature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D
Afeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02C
Afeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpµ
2feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02H
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp£
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1Ð
Wfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02Y
Wfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpÖ
Yfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02[
Yfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1â
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3Ï
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValueAssignVariableOp`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceUfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0X^feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*s
_classi
geloc:@feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02H
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValueÝ
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1AssignVariableOpbfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceYfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0Z^feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*u
_classk
igloc:@feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1ñ
.feature_extractor_cnn/cnn_block_6/re_lu_6/ReluReluLfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.feature_extractor_cnn/cnn_block_6/re_lu_6/Reluß
Efeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2G
Efeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices³
3feature_extractor_cnn/global_average_pooling2d/MeanMean<feature_extractor_cnn/cnn_block_6/re_lu_6/Relu:activations:0Nfeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3feature_extractor_cnn/global_average_pooling2d/MeanÇ
(feed_forward/dense/MatMul/ReadVariableOpReadVariableOp1feed_forward_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02*
(feed_forward/dense/MatMul/ReadVariableOpâ
feed_forward/dense/MatMulMatMul<feature_extractor_cnn/global_average_pooling2d/Mean:output:00feed_forward/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/MatMulÅ
)feed_forward/dense/BiasAdd/ReadVariableOpReadVariableOp2feed_forward_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)feed_forward/dense/BiasAdd/ReadVariableOpÍ
feed_forward/dense/BiasAddBiasAdd#feed_forward/dense/MatMul:product:01feed_forward/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/BiasAdd
feed_forward/dense/ReluRelu#feed_forward/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/Relu
"feed_forward/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"feed_forward/dropout/dropout/ConstÑ
 feed_forward/dropout/dropout/MulMul%feed_forward/dense/Relu:activations:0+feed_forward/dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 feed_forward/dropout/dropout/Mul
"feed_forward/dropout/dropout/ShapeShape%feed_forward/dense/Relu:activations:0*
T0*
_output_shapes
:2$
"feed_forward/dropout/dropout/Shapeó
9feed_forward/dropout/dropout/random_uniform/RandomUniformRandomUniform+feed_forward/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02;
9feed_forward/dropout/dropout/random_uniform/RandomUniform
+feed_forward/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2-
+feed_forward/dropout/dropout/GreaterEqual/y
)feed_forward/dropout/dropout/GreaterEqualGreaterEqualBfeed_forward/dropout/dropout/random_uniform/RandomUniform:output:04feed_forward/dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)feed_forward/dropout/dropout/GreaterEqual¾
!feed_forward/dropout/dropout/CastCast-feed_forward/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!feed_forward/dropout/dropout/CastÎ
"feed_forward/dropout/dropout/Mul_1Mul$feed_forward/dropout/dropout/Mul:z:0%feed_forward/dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"feed_forward/dropout/dropout/Mul_1Ì
*feed_forward/dense_1/MatMul/ReadVariableOpReadVariableOp3feed_forward_dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02,
*feed_forward/dense_1/MatMul/ReadVariableOpÒ
feed_forward/dense_1/MatMulMatMul&feed_forward/dropout/dropout/Mul_1:z:02feed_forward/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
feed_forward/dense_1/MatMulË
+feed_forward/dense_1/BiasAdd/ReadVariableOpReadVariableOp4feed_forward_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+feed_forward/dense_1/BiasAdd/ReadVariableOpÕ
feed_forward/dense_1/BiasAddBiasAdd%feed_forward/dense_1/MatMul:product:03feed_forward/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
feed_forward/dense_1/BiasAdd	
IdentityIdentity%feed_forward/dense_1/BiasAdd:output:0E^feature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValueG^feature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1G^feature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValueI^feature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1G^feature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValueI^feature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1G^feature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValueI^feature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1G^feature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValueI^feature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1G^feature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValueI^feature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1G^feature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValueI^feature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::2
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValueDfeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue2
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_1Ffeature_extractor_cnn/cnn_block_0/batch_normalization/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValueFfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValueFfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValueFfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValueFfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValueFfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/AssignNewValue_12
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValueFfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue2
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967252

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
 

H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6965984
conv2d_2_input+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_2/Conv2D/ReadVariableOpÇ
conv2d_2/Conv2DConv2Dconv2d_2_input&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_2/BiasAdd¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_2/FusedBatchNormV3
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_2/ReluÆ
IdentityIdentityre_lu_2/Relu:activations:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_2_input
ª
ª
7__inference_batch_normalization_1_layer_call_fn_6967044

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_69596172
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

´
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966267

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp¿
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp¬
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_3/BiasAdd¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_3/Reluv
IdentityIdentityre_lu_3/Relu:activations:0*
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

¯
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6960647

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
¬
ª
7__inference_batch_normalization_2_layer_call_fn_6967214

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_69599612
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
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6960925

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

¯
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967705

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
Ò
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6967283

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
Á
E
)__inference_re_lu_6_layer_call_fn_6967916

inputs
identityÖ
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
   E8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_69613322
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
Å
ò
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6961105

inputs
conv2d_5_6961089
conv2d_5_6961091!
batch_normalization_5_6961094!
batch_normalization_5_6961096!
batch_normalization_5_6961098!
batch_normalization_5_6961100
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCallª
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_6961089conv2d_5_6961091*
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
   E8 *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_69609252"
 conv2d_5/StatefulPartitionedCallÐ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_6961094batch_normalization_5_6961096batch_normalization_5_6961098batch_normalization_5_6961100*
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
   E8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_69609782/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
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
   E8 *M
fHRF
D__inference_re_lu_5_layer_call_and_return_conditional_losses_69610192
re_lu_5/PartitionedCallÏ
IdentityIdentity re_lu_5/PartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
ª
7__inference_batch_normalization_3_layer_call_fn_6967422

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
:ÿÿÿÿÿÿÿÿÿ *$
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_69603342
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
§
­
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6967455

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
µø
Ì
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6964580

inputs5
1cnn_block_0_conv2d_conv2d_readvariableop_resource6
2cnn_block_0_conv2d_biasadd_readvariableop_resource;
7cnn_block_0_batch_normalization_readvariableop_resource=
9cnn_block_0_batch_normalization_readvariableop_1_resourceL
Hcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourceN
Jcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_1_conv2d_1_conv2d_readvariableop_resource8
4cnn_block_1_conv2d_1_biasadd_readvariableop_resource=
9cnn_block_1_batch_normalization_1_readvariableop_resource?
;cnn_block_1_batch_normalization_1_readvariableop_1_resourceN
Jcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_2_conv2d_2_conv2d_readvariableop_resource8
4cnn_block_2_conv2d_2_biasadd_readvariableop_resource=
9cnn_block_2_batch_normalization_2_readvariableop_resource?
;cnn_block_2_batch_normalization_2_readvariableop_1_resourceN
Jcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource8
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource=
9cnn_block_3_batch_normalization_3_readvariableop_resource?
;cnn_block_3_batch_normalization_3_readvariableop_1_resourceN
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource8
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource=
9cnn_block_4_batch_normalization_4_readvariableop_resource?
;cnn_block_4_batch_normalization_4_readvariableop_1_resourceN
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource8
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource=
9cnn_block_5_batch_normalization_5_readvariableop_resource?
;cnn_block_5_batch_normalization_5_readvariableop_1_resourceN
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_6_conv2d_6_conv2d_readvariableop_resource8
4cnn_block_6_conv2d_6_biasadd_readvariableop_resource=
9cnn_block_6_batch_normalization_6_readvariableop_resource?
;cnn_block_6_batch_normalization_6_readvariableop_1_resourceN
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity¢.cnn_block_0/batch_normalization/AssignNewValue¢0cnn_block_0/batch_normalization/AssignNewValue_1¢0cnn_block_1/batch_normalization_1/AssignNewValue¢2cnn_block_1/batch_normalization_1/AssignNewValue_1¢0cnn_block_2/batch_normalization_2/AssignNewValue¢2cnn_block_2/batch_normalization_2/AssignNewValue_1¢0cnn_block_3/batch_normalization_3/AssignNewValue¢2cnn_block_3/batch_normalization_3/AssignNewValue_1¢0cnn_block_4/batch_normalization_4/AssignNewValue¢2cnn_block_4/batch_normalization_4/AssignNewValue_1¢0cnn_block_5/batch_normalization_5/AssignNewValue¢2cnn_block_5/batch_normalization_5/AssignNewValue_1¢0cnn_block_6/batch_normalization_6/AssignNewValue¢2cnn_block_6/batch_normalization_6/AssignNewValue_1Î
(cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOp1cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(cnn_block_0/conv2d/Conv2D/ReadVariableOpÝ
cnn_block_0/conv2d/Conv2DConv2Dinputs0cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_0/conv2d/Conv2DÅ
)cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOp2cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)cnn_block_0/conv2d/BiasAdd/ReadVariableOpÔ
cnn_block_0/conv2d/BiasAddBiasAdd"cnn_block_0/conv2d/Conv2D:output:01cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/conv2d/BiasAddÔ
.cnn_block_0/batch_normalization/ReadVariableOpReadVariableOp7cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype020
.cnn_block_0/batch_normalization/ReadVariableOpÚ
0cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOp9cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype022
0cnn_block_0/batch_normalization/ReadVariableOp_1
?cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp
Acnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Acnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1µ
0cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV3#cnn_block_0/conv2d/BiasAdd:output:06cnn_block_0/batch_normalization/ReadVariableOp:value:08cnn_block_0/batch_normalization/ReadVariableOp_1:value:0Gcnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Icnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<22
0cnn_block_0/batch_normalization/FusedBatchNormV3¿
.cnn_block_0/batch_normalization/AssignNewValueAssignVariableOpHcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource=cnn_block_0/batch_normalization/FusedBatchNormV3:batch_mean:0@^cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp*[
_classQ
OMloc:@cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype020
.cnn_block_0/batch_normalization/AssignNewValueÍ
0cnn_block_0/batch_normalization/AssignNewValue_1AssignVariableOpJcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceAcnn_block_0/batch_normalization/FusedBatchNormV3:batch_variance:0B^cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*]
_classS
QOloc:@cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype022
0cnn_block_0/batch_normalization/AssignNewValue_1¨
cnn_block_0/re_lu/ReluRelu4cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/re_lu/ReluÔ
*cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_1/conv2d_1/Conv2D/ReadVariableOp
cnn_block_1/conv2d_1/Conv2DConv2D$cnn_block_0/re_lu/Relu:activations:02cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_1/conv2d_1/Conv2DË
+cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpÜ
cnn_block_1/conv2d_1/BiasAddBiasAdd$cnn_block_1/conv2d_1/Conv2D:output:03cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/conv2d_1/BiasAddÚ
0cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOp9cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_1/batch_normalization_1/ReadVariableOpà
2cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp;cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_1/batch_normalization_1/ReadVariableOp_1
Acnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Ccnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%cnn_block_1/conv2d_1/BiasAdd:output:08cnn_block_1/batch_normalization_1/ReadVariableOp:value:0:cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0Icnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_1/batch_normalization_1/FusedBatchNormV3Ë
0cnn_block_1/batch_normalization_1/AssignNewValueAssignVariableOpJcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource?cnn_block_1/batch_normalization_1/FusedBatchNormV3:batch_mean:0B^cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_1/batch_normalization_1/AssignNewValueÙ
2cnn_block_1/batch_normalization_1/AssignNewValue_1AssignVariableOpLcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_1/batch_normalization_1/FusedBatchNormV3:batch_variance:0D^cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_1/batch_normalization_1/AssignNewValue_1®
cnn_block_1/re_lu_1/ReluRelu6cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/re_lu_1/ReluÔ
*cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_2/conv2d_2/Conv2D/ReadVariableOp
cnn_block_2/conv2d_2/Conv2DConv2D&cnn_block_1/re_lu_1/Relu:activations:02cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_2/conv2d_2/Conv2DË
+cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpÜ
cnn_block_2/conv2d_2/BiasAddBiasAdd$cnn_block_2/conv2d_2/Conv2D:output:03cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/conv2d_2/BiasAddÚ
0cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOp9cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_2/batch_normalization_2/ReadVariableOpà
2cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp;cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_2/batch_normalization_2/ReadVariableOp_1
Acnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Ccnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%cnn_block_2/conv2d_2/BiasAdd:output:08cnn_block_2/batch_normalization_2/ReadVariableOp:value:0:cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0Icnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_2/batch_normalization_2/FusedBatchNormV3Ë
0cnn_block_2/batch_normalization_2/AssignNewValueAssignVariableOpJcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource?cnn_block_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0B^cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_2/batch_normalization_2/AssignNewValueÙ
2cnn_block_2/batch_normalization_2/AssignNewValue_1AssignVariableOpLcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0D^cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_2/batch_normalization_2/AssignNewValue_1®
cnn_block_2/re_lu_2/ReluRelu6cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/re_lu_2/ReluÔ
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp
cnn_block_3/conv2d_3/Conv2DConv2D&cnn_block_2/re_lu_2/Relu:activations:02cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_3/conv2d_3/Conv2DË
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpÜ
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/conv2d_3/BiasAddÚ
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_3/batch_normalization_3/ReadVariableOpà
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_3/batch_normalization_3/ReadVariableOp_1
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_3/batch_normalization_3/FusedBatchNormV3Ë
0cnn_block_3/batch_normalization_3/AssignNewValueAssignVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?cnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_3/batch_normalization_3/AssignNewValueÙ
2cnn_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_3/batch_normalization_3/AssignNewValue_1®
cnn_block_3/re_lu_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/re_lu_3/ReluÔ
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp
cnn_block_4/conv2d_4/Conv2DConv2D&cnn_block_3/re_lu_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_4/conv2d_4/Conv2DË
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpÜ
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/conv2d_4/BiasAddÚ
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_4/batch_normalization_4/ReadVariableOpà
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_4/batch_normalization_4/ReadVariableOp_1
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_4/batch_normalization_4/FusedBatchNormV3Ë
0cnn_block_4/batch_normalization_4/AssignNewValueAssignVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource?cnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0B^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_4/batch_normalization_4/AssignNewValueÙ
2cnn_block_4/batch_normalization_4/AssignNewValue_1AssignVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0D^cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_4/batch_normalization_4/AssignNewValue_1®
cnn_block_4/re_lu_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/re_lu_4/ReluÔ
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp
cnn_block_5/conv2d_5/Conv2DConv2D&cnn_block_4/re_lu_4/Relu:activations:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_5/conv2d_5/Conv2DË
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpÜ
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/conv2d_5/BiasAddÚ
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_5/batch_normalization_5/ReadVariableOpà
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_5/batch_normalization_5/ReadVariableOp_1
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ã
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_5/batch_normalization_5/FusedBatchNormV3Ë
0cnn_block_5/batch_normalization_5/AssignNewValueAssignVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?cnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_5/batch_normalization_5/AssignNewValueÙ
2cnn_block_5/batch_normalization_5/AssignNewValue_1AssignVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_5/batch_normalization_5/AssignNewValue_1®
cnn_block_5/re_lu_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/re_lu_5/ReluÕ
*cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*cnn_block_6/conv2d_6/Conv2D/ReadVariableOp
cnn_block_6/conv2d_6/Conv2DConv2D&cnn_block_5/re_lu_5/Relu:activations:02cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_6/conv2d_6/Conv2DÌ
+cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpÝ
cnn_block_6/conv2d_6/BiasAddBiasAdd$cnn_block_6/conv2d_6/Conv2D:output:03cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/conv2d_6/BiasAddÛ
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype022
0cnn_block_6/batch_normalization_6/ReadVariableOpá
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2cnn_block_6/batch_normalization_6/ReadVariableOp_1
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1È
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_6/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<24
2cnn_block_6/batch_normalization_6/FusedBatchNormV3Ë
0cnn_block_6/batch_normalization_6/AssignNewValueAssignVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource?cnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_mean:0B^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0cnn_block_6/batch_normalization_6/AssignNewValueÙ
2cnn_block_6/batch_normalization_6/AssignNewValue_1AssignVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceCcnn_block_6/batch_normalization_6/FusedBatchNormV3:batch_variance:0D^cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2cnn_block_6/batch_normalization_6/AssignNewValue_1¯
cnn_block_6/re_lu_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/re_lu_6/Relu³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesÛ
global_average_pooling2d/MeanMean&cnn_block_6/re_lu_6/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_average_pooling2d/MeanÏ
IdentityIdentity&global_average_pooling2d/Mean:output:0/^cnn_block_0/batch_normalization/AssignNewValue1^cnn_block_0/batch_normalization/AssignNewValue_11^cnn_block_1/batch_normalization_1/AssignNewValue3^cnn_block_1/batch_normalization_1/AssignNewValue_11^cnn_block_2/batch_normalization_2/AssignNewValue3^cnn_block_2/batch_normalization_2/AssignNewValue_11^cnn_block_3/batch_normalization_3/AssignNewValue3^cnn_block_3/batch_normalization_3/AssignNewValue_11^cnn_block_4/batch_normalization_4/AssignNewValue3^cnn_block_4/batch_normalization_4/AssignNewValue_11^cnn_block_5/batch_normalization_5/AssignNewValue3^cnn_block_5/batch_normalization_5/AssignNewValue_11^cnn_block_6/batch_normalization_6/AssignNewValue3^cnn_block_6/batch_normalization_6/AssignNewValue_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ø
_input_shapesÆ
Ã:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::2`
.cnn_block_0/batch_normalization/AssignNewValue.cnn_block_0/batch_normalization/AssignNewValue2d
0cnn_block_0/batch_normalization/AssignNewValue_10cnn_block_0/batch_normalization/AssignNewValue_12d
0cnn_block_1/batch_normalization_1/AssignNewValue0cnn_block_1/batch_normalization_1/AssignNewValue2h
2cnn_block_1/batch_normalization_1/AssignNewValue_12cnn_block_1/batch_normalization_1/AssignNewValue_12d
0cnn_block_2/batch_normalization_2/AssignNewValue0cnn_block_2/batch_normalization_2/AssignNewValue2h
2cnn_block_2/batch_normalization_2/AssignNewValue_12cnn_block_2/batch_normalization_2/AssignNewValue_12d
0cnn_block_3/batch_normalization_3/AssignNewValue0cnn_block_3/batch_normalization_3/AssignNewValue2h
2cnn_block_3/batch_normalization_3/AssignNewValue_12cnn_block_3/batch_normalization_3/AssignNewValue_12d
0cnn_block_4/batch_normalization_4/AssignNewValue0cnn_block_4/batch_normalization_4/AssignNewValue2h
2cnn_block_4/batch_normalization_4/AssignNewValue_12cnn_block_4/batch_normalization_4/AssignNewValue_12d
0cnn_block_5/batch_normalization_5/AssignNewValue0cnn_block_5/batch_normalization_5/AssignNewValue2h
2cnn_block_5/batch_normalization_5/AssignNewValue_12cnn_block_5/batch_normalization_5/AssignNewValue_12d
0cnn_block_6/batch_normalization_6/AssignNewValue0cnn_block_6/batch_normalization_6/AssignNewValue2h
2cnn_block_6/batch_normalization_6/AssignNewValue_12cnn_block_6/batch_normalization_6/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
ò
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6959853

inputs
conv2d_1_6959837
conv2d_1_6959839!
batch_normalization_1_6959842!
batch_normalization_1_6959844!
batch_normalization_1_6959846!
batch_normalization_1_6959848
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCallª
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_6959837conv2d_1_6959839*
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
E__inference_conv2d_1_layer_call_and_return_conditional_losses_69596732"
 conv2d_1/StatefulPartitionedCallÐ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_6959842batch_normalization_1_6959844batch_normalization_1_6959846batch_normalization_1_6959848*
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_69597262/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_1_layer_call_and_return_conditional_losses_69597672
re_lu_1/PartitionedCallÏ
IdentityIdentity re_lu_1/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
-__inference_cnn_block_4_layer_call_fn_6966387

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
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69607922
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
Ñ

R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6960665

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

E
)__inference_dropout_layer_call_fn_6965594

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
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
D__inference_dropout_layer_call_and_return_conditional_losses_69622962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª
Æ
-__inference_cnn_block_6_layer_call_fn_6966731
conv2d_6_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69614182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_6_input


I__inference_feed_forward_layer_call_and_return_conditional_losses_6965521

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Identity¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
ò
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6960166

inputs
conv2d_2_6960150
conv2d_2_6960152!
batch_normalization_2_6960155!
batch_normalization_2_6960157!
batch_normalization_2_6960159!
batch_normalization_2_6960161
identity¢-batch_normalization_2/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCallª
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_6960150conv2d_2_6960152*
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
E__inference_conv2d_2_layer_call_and_return_conditional_losses_69599862"
 conv2d_2/StatefulPartitionedCallÐ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_6960155batch_normalization_2_6960157batch_normalization_2_6960159batch_normalization_2_6960161*
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_69600392/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_2_layer_call_and_return_conditional_losses_69600802
re_lu_2/PartitionedCallÏ
IdentityIdentity re_lu_2/PartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨
Æ
-__inference_cnn_block_2_layer_call_fn_6966043
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69601662
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
_user_specified_nameconv2d_2_input
Ê
¯
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6960243

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
¹
C
'__inference_re_lu_layer_call_fn_6966974

inputs
identityÓ
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
   E8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_69594542
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
Ã
ò
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6960443

inputs
conv2d_3_6960427
conv2d_3_6960429!
batch_normalization_3_6960432!
batch_normalization_3_6960434!
batch_normalization_3_6960436!
batch_normalization_3_6960438
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCallª
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_6960427conv2d_3_6960429*
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
   E8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_69602992"
 conv2d_3/StatefulPartitionedCallÎ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_6960432batch_normalization_3_6960434batch_normalization_3_6960436batch_normalization_3_6960438*
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
   E8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_69603342/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
   E8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_69603932
re_lu_3/PartitionedCallÏ
IdentityIdentity re_lu_3/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

c
D__inference_dropout_layer_call_and_return_conditional_losses_6962291

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§
­
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6967612

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
à
¨
5__inference_batch_normalization_layer_call_fn_6966964

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
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_69594132
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
¬¼

#__inference__traced_restore_6968225
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta&
"assignvariableop_4_conv2d_1_kernel$
 assignvariableop_5_conv2d_1_bias2
.assignvariableop_6_batch_normalization_1_gamma1
-assignvariableop_7_batch_normalization_1_beta&
"assignvariableop_8_conv2d_2_kernel$
 assignvariableop_9_conv2d_2_bias3
/assignvariableop_10_batch_normalization_2_gamma2
.assignvariableop_11_batch_normalization_2_beta'
#assignvariableop_12_conv2d_3_kernel%
!assignvariableop_13_conv2d_3_bias3
/assignvariableop_14_batch_normalization_3_gamma2
.assignvariableop_15_batch_normalization_3_beta'
#assignvariableop_16_conv2d_4_kernel%
!assignvariableop_17_conv2d_4_bias3
/assignvariableop_18_batch_normalization_4_gamma2
.assignvariableop_19_batch_normalization_4_beta'
#assignvariableop_20_conv2d_5_kernel%
!assignvariableop_21_conv2d_5_bias3
/assignvariableop_22_batch_normalization_5_gamma2
.assignvariableop_23_batch_normalization_5_beta'
#assignvariableop_24_conv2d_6_kernel%
!assignvariableop_25_conv2d_6_bias3
/assignvariableop_26_batch_normalization_6_gamma2
.assignvariableop_27_batch_normalization_6_beta7
3assignvariableop_28_batch_normalization_moving_mean;
7assignvariableop_29_batch_normalization_moving_variance9
5assignvariableop_30_batch_normalization_1_moving_mean=
9assignvariableop_31_batch_normalization_1_moving_variance9
5assignvariableop_32_batch_normalization_2_moving_mean=
9assignvariableop_33_batch_normalization_2_moving_variance9
5assignvariableop_34_batch_normalization_3_moving_mean=
9assignvariableop_35_batch_normalization_3_moving_variance9
5assignvariableop_36_batch_normalization_4_moving_mean=
9assignvariableop_37_batch_normalization_4_moving_variance9
5assignvariableop_38_batch_normalization_5_moving_mean=
9assignvariableop_39_batch_normalization_5_moving_variance9
5assignvariableop_40_batch_normalization_6_moving_mean=
9assignvariableop_41_batch_normalization_6_moving_variance$
 assignvariableop_42_dense_kernel"
assignvariableop_43_dense_bias&
"assignvariableop_44_dense_1_kernel$
 assignvariableop_45_dense_1_bias
identity_47¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9û
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*
valueýBú/B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesì
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2±
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3°
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6³
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7²
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14·
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¶
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_4_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18·
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_4_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¶
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_4_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20«
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_5_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22·
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_5_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¶
AssignVariableOp_23AssignVariableOp.assignvariableop_23_batch_normalization_5_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24«
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_6_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25©
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_6_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_6_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¶
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_6_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28»
AssignVariableOp_28AssignVariableOp3assignvariableop_28_batch_normalization_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¿
AssignVariableOp_29AssignVariableOp7assignvariableop_29_batch_normalization_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30½
AssignVariableOp_30AssignVariableOp5assignvariableop_30_batch_normalization_1_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Á
AssignVariableOp_31AssignVariableOp9assignvariableop_31_batch_normalization_1_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32½
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_2_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Á
AssignVariableOp_33AssignVariableOp9assignvariableop_33_batch_normalization_2_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34½
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_3_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Á
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_3_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36½
AssignVariableOp_36AssignVariableOp5assignvariableop_36_batch_normalization_4_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Á
AssignVariableOp_37AssignVariableOp9assignvariableop_37_batch_normalization_4_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38½
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_5_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Á
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_5_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40½
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_6_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Á
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_6_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¨
AssignVariableOp_42AssignVariableOp assignvariableop_42_dense_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¦
AssignVariableOp_43AssignVariableOpassignvariableop_43_dense_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44ª
AssignVariableOp_44AssignVariableOp"assignvariableop_44_dense_1_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¨
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_1_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_459
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÒ
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_46Å
Identity_47IdentityIdentity_46:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_47"#
identity_47Identity_47:output:0*Ï
_input_shapes½
º: ::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
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
§
­
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6959673

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

¾
-__inference_cnn_block_4_layer_call_fn_6966370

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
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69607562
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
È
­
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966856

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
Í
¬
D__inference_dense_1_layer_call_and_return_conditional_losses_6962319

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967077

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
Ã
ò
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6959817

inputs
conv2d_1_6959801
conv2d_1_6959803!
batch_normalization_1_6959806!
batch_normalization_1_6959808!
batch_normalization_1_6959810!
batch_normalization_1_6959812
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCallª
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_6959801conv2d_1_6959803*
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
E__inference_conv2d_1_layer_call_and_return_conditional_losses_69596732"
 conv2d_1/StatefulPartitionedCallÎ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_6959806batch_normalization_1_6959808batch_normalization_1_6959810batch_normalization_1_6959812*
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_69597082/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
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
D__inference_re_lu_1_layer_call_and_return_conditional_losses_69597672
re_lu_1/PartitionedCallÏ
IdentityIdentity re_lu_1/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
|
'__inference_dense_layer_call_fn_6965567

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_69622632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð'
â
F__inference_class_cnn_layer_call_and_return_conditional_losses_6963113

inputs!
feature_extractor_cnn_6963018!
feature_extractor_cnn_6963020!
feature_extractor_cnn_6963022!
feature_extractor_cnn_6963024!
feature_extractor_cnn_6963026!
feature_extractor_cnn_6963028!
feature_extractor_cnn_6963030!
feature_extractor_cnn_6963032!
feature_extractor_cnn_6963034!
feature_extractor_cnn_6963036!
feature_extractor_cnn_6963038!
feature_extractor_cnn_6963040!
feature_extractor_cnn_6963042!
feature_extractor_cnn_6963044!
feature_extractor_cnn_6963046!
feature_extractor_cnn_6963048!
feature_extractor_cnn_6963050!
feature_extractor_cnn_6963052!
feature_extractor_cnn_6963054!
feature_extractor_cnn_6963056!
feature_extractor_cnn_6963058!
feature_extractor_cnn_6963060!
feature_extractor_cnn_6963062!
feature_extractor_cnn_6963064!
feature_extractor_cnn_6963066!
feature_extractor_cnn_6963068!
feature_extractor_cnn_6963070!
feature_extractor_cnn_6963072!
feature_extractor_cnn_6963074!
feature_extractor_cnn_6963076!
feature_extractor_cnn_6963078!
feature_extractor_cnn_6963080!
feature_extractor_cnn_6963082!
feature_extractor_cnn_6963084!
feature_extractor_cnn_6963086!
feature_extractor_cnn_6963088!
feature_extractor_cnn_6963090!
feature_extractor_cnn_6963092!
feature_extractor_cnn_6963094!
feature_extractor_cnn_6963096!
feature_extractor_cnn_6963098!
feature_extractor_cnn_6963100
feed_forward_6963103
feed_forward_6963105
feed_forward_6963107
feed_forward_6963109
identity¢-feature_extractor_cnn/StatefulPartitionedCall¢$feed_forward/StatefulPartitionedCall[
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
truediv
-feature_extractor_cnn/StatefulPartitionedCallStatefulPartitionedCalltruediv:z:0feature_extractor_cnn_6963018feature_extractor_cnn_6963020feature_extractor_cnn_6963022feature_extractor_cnn_6963024feature_extractor_cnn_6963026feature_extractor_cnn_6963028feature_extractor_cnn_6963030feature_extractor_cnn_6963032feature_extractor_cnn_6963034feature_extractor_cnn_6963036feature_extractor_cnn_6963038feature_extractor_cnn_6963040feature_extractor_cnn_6963042feature_extractor_cnn_6963044feature_extractor_cnn_6963046feature_extractor_cnn_6963048feature_extractor_cnn_6963050feature_extractor_cnn_6963052feature_extractor_cnn_6963054feature_extractor_cnn_6963056feature_extractor_cnn_6963058feature_extractor_cnn_6963060feature_extractor_cnn_6963062feature_extractor_cnn_6963064feature_extractor_cnn_6963066feature_extractor_cnn_6963068feature_extractor_cnn_6963070feature_extractor_cnn_6963072feature_extractor_cnn_6963074feature_extractor_cnn_6963076feature_extractor_cnn_6963078feature_extractor_cnn_6963080feature_extractor_cnn_6963082feature_extractor_cnn_6963084feature_extractor_cnn_6963086feature_extractor_cnn_6963088feature_extractor_cnn_6963090feature_extractor_cnn_6963092feature_extractor_cnn_6963094feature_extractor_cnn_6963096feature_extractor_cnn_6963098feature_extractor_cnn_6963100*6
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
   E8 *[
fVRT
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_69621612/
-feature_extractor_cnn/StatefulPartitionedCall
$feed_forward/StatefulPartitionedCallStatefulPartitionedCall6feature_extractor_cnn/StatefulPartitionedCall:output:0feed_forward_6963103feed_forward_6963105feed_forward_6963107feed_forward_6963109*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *R
fMRK
I__inference_feed_forward_layer_call_and_return_conditional_losses_69623972&
$feed_forward/StatefulPartitionedCallØ
IdentityIdentity-feed_forward/StatefulPartitionedCall:output:0.^feature_extractor_cnn/StatefulPartitionedCall%^feed_forward/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::2^
-feature_extractor_cnn/StatefulPartitionedCall-feature_extractor_cnn/StatefulPartitionedCall2L
$feed_forward/StatefulPartitionedCall$feed_forward/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
¸
+__inference_class_cnn_layer_call_fn_6963860
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

unknown_41

unknown_42

unknown_43

unknown_44
identity¢StatefulPartitionedCallá
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
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_class_cnn_layer_call_and_return_conditional_losses_69631132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ä
ª
7__inference_batch_normalization_2_layer_call_fn_6967278

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_69600392
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
ª
ª
7__inference_batch_normalization_3_layer_call_fn_6967358

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_69602432
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
É
ò
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6961418

inputs
conv2d_6_6961402
conv2d_6_6961404!
batch_normalization_6_6961407!
batch_normalization_6_6961409!
batch_normalization_6_6961411!
batch_normalization_6_6961413
identity¢-batch_normalization_6/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall«
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_6961402conv2d_6_6961404*
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
   E8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_69612382"
 conv2d_6/StatefulPartitionedCallÑ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_6961407batch_normalization_6_6961409batch_normalization_6_6961411batch_normalization_6_6961413*
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
   E8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69612912/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
   E8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_69613322
re_lu_6/PartitionedCallÐ
IdentityIdentity re_lu_6/PartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966156
conv2d_3_input+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOpÇ
conv2d_3/Conv2DConv2Dconv2d_3_input&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp¬
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_3/BiasAdd¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_3/FusedBatchNormV3
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_3/ReluÆ
IdentityIdentityre_lu_3/Relu:activations:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_3_input


*__inference_conv2d_1_layer_call_fn_6966993

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
E__inference_conv2d_1_layer_call_and_return_conditional_losses_69596732
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
Ö
¯
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967798

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
½
E
)__inference_re_lu_2_layer_call_fn_6967288

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
D__inference_re_lu_2_layer_call_and_return_conditional_losses_69600802
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
â
ª
7__inference_batch_normalization_4_layer_call_fn_6967515

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
:ÿÿÿÿÿÿÿÿÿ@*$
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_69606472
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
Â
®
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965751
conv2d_input)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identityª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp¿
conv2d/Conv2DConv2Dconv2d_input$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ó
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

re_lu/Relut
IdentityIdentityre_lu/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:::::::] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
ä
ª
7__inference_batch_normalization_5_layer_call_fn_6967749

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
:ÿÿÿÿÿÿÿÿÿ@*&
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_69609782
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
¬ô
ê
F__inference_class_cnn_layer_call_and_return_conditional_losses_6964219

inputsK
Gfeature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resourceL
Hfeature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resourceQ
Mfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resourceb
^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourced
`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceM
Ifeature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resourceN
Jfeature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resourceS
Ofeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resourceU
Qfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resourced
`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourcef
bfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource5
1feed_forward_dense_matmul_readvariableop_resource6
2feed_forward_dense_biasadd_readvariableop_resource7
3feed_forward_dense_1_matmul_readvariableop_resource8
4feed_forward_dense_1_biasadd_readvariableop_resource
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
truediv
>feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOpGfeature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02@
>feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp¤
/feature_extractor_cnn/cnn_block_0/conv2d/Conv2DConv2Dtruediv:z:0Ffeature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
21
/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D
?feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOpHfeature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp¬
0feature_extractor_cnn/cnn_block_0/conv2d/BiasAddBiasAdd8feature_extractor_cnn/cnn_block_0/conv2d/Conv2D:output:0Gfeature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOpReadVariableOpMfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02F
Dfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOpOfeature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1É
Ufeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp^feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02W
Ufeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpÏ
Wfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp`feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02Y
Wfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Á
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV39feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd:output:0Lfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp:value:0Nfeature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1:value:0]feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0_feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2H
Ffeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3ê
,feature_extractor_cnn/cnn_block_0/re_lu/ReluReluJfeature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,feature_extractor_cnn/cnn_block_0/re_lu/Relu
@feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02B
@feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpÙ
1feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2DConv2D:feature_extractor_cnn/cnn_block_0/re_lu/Relu:activations:0Hfeature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D
Afeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_1/re_lu_1/ReluReluLfeature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_1/re_lu_1/Relu
@feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02B
@feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2DConv2D<feature_extractor_cnn/cnn_block_1/re_lu_1/Relu:activations:0Hfeature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D
Afeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_2/re_lu_2/ReluReluLfeature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_2/re_lu_2/Relu
@feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02B
@feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2DConv2D<feature_extractor_cnn/cnn_block_2/re_lu_2/Relu:activations:0Hfeature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D
Afeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Afeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02H
Ffeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02[
Yfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_3/re_lu_3/ReluReluLfeature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.feature_extractor_cnn/cnn_block_3/re_lu_3/Relu
@feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02B
@feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2DConv2D<feature_extractor_cnn/cnn_block_3/re_lu_3/Relu:activations:0Hfeature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D
Afeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Afeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02H
Ffeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Y
Wfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02[
Yfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_4/re_lu_4/ReluReluLfeature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.feature_extractor_cnn/cnn_block_4/re_lu_4/Relu
@feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02B
@feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpÛ
1feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2DConv2D<feature_extractor_cnn/cnn_block_4/re_lu_4/Relu:activations:0Hfeature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D
Afeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Afeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp´
2feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@24
2feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02H
Ffeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp¢
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1Ï
Wfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Y
Wfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpÕ
Yfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02[
Yfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ï
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3ð
.feature_extractor_cnn/cnn_block_5/re_lu_5/ReluReluLfeature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.feature_extractor_cnn/cnn_block_5/re_lu_5/Relu
@feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpIfeature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02B
@feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpÜ
1feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2DConv2D<feature_extractor_cnn/cnn_block_5/re_lu_5/Relu:activations:0Hfeature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
23
1feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D
Afeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpJfeature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02C
Afeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpµ
2feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAddBiasAdd:feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D:output:0Ifeature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpOfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02H
Ffeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp£
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOpQfeature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1Ð
Wfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp`feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02Y
Wfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpÖ
Yfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbfeature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02[
Yfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ô
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3;feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd:output:0Nfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Pfeature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0_feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0afeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2J
Hfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3ñ
.feature_extractor_cnn/cnn_block_6/re_lu_6/ReluReluLfeature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.feature_extractor_cnn/cnn_block_6/re_lu_6/Reluß
Efeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2G
Efeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices³
3feature_extractor_cnn/global_average_pooling2d/MeanMean<feature_extractor_cnn/cnn_block_6/re_lu_6/Relu:activations:0Nfeature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3feature_extractor_cnn/global_average_pooling2d/MeanÇ
(feed_forward/dense/MatMul/ReadVariableOpReadVariableOp1feed_forward_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02*
(feed_forward/dense/MatMul/ReadVariableOpâ
feed_forward/dense/MatMulMatMul<feature_extractor_cnn/global_average_pooling2d/Mean:output:00feed_forward/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/MatMulÅ
)feed_forward/dense/BiasAdd/ReadVariableOpReadVariableOp2feed_forward_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)feed_forward/dense/BiasAdd/ReadVariableOpÍ
feed_forward/dense/BiasAddBiasAdd#feed_forward/dense/MatMul:product:01feed_forward/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/BiasAdd
feed_forward/dense/ReluRelu#feed_forward/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dense/Relu£
feed_forward/dropout/IdentityIdentity%feed_forward/dense/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
feed_forward/dropout/IdentityÌ
*feed_forward/dense_1/MatMul/ReadVariableOpReadVariableOp3feed_forward_dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02,
*feed_forward/dense_1/MatMul/ReadVariableOpÒ
feed_forward/dense_1/MatMulMatMul&feed_forward/dropout/Identity:output:02feed_forward/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
feed_forward/dense_1/MatMulË
+feed_forward/dense_1/BiasAdd/ReadVariableOpReadVariableOp4feed_forward_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+feed_forward/dense_1/BiasAdd/ReadVariableOpÕ
feed_forward/dense_1/BiasAddBiasAdd%feed_forward/dense_1/MatMul:product:03feed_forward/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
feed_forward/dense_1/BiasAddy
IdentityIdentity%feed_forward/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
ª
7__inference_batch_normalization_5_layer_call_fn_6967736

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
:ÿÿÿÿÿÿÿÿÿ@*$
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_69609602
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
ä

I__inference_feed_forward_layer_call_and_return_conditional_losses_6962369

inputs
dense_6962357
dense_6962359
dense_1_6962363
dense_1_6962365
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6962357dense_6962359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_69622632
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
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
D__inference_dropout_layer_call_and_return_conditional_losses_69622912!
dropout/StatefulPartitionedCall¿
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_6962363dense_1_6962365*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_69623192!
dense_1/StatefulPartitionedCallà
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
ª
B__inference_dense_layer_call_and_return_conditional_losses_6962263

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
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
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966009
conv2d_2_input+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_2/Conv2D/ReadVariableOpÇ
conv2d_2/Conv2DConv2Dconv2d_2_input&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_2/BiasAdd¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_2/Reluv
IdentityIdentityre_lu_2/Relu:activations:0*
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
_user_specified_nameconv2d_2_input
ª
ª
7__inference_batch_normalization_5_layer_call_fn_6967672

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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_69608692
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
ñ
ô
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965640

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1ª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1á
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2&
$batch_normalization/FusedBatchNormV3÷
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

re_lu/ReluÀ
IdentityIdentityre_lu/Relu:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967816

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
 
Ä
-__inference_cnn_block_0_layer_call_fn_6965768
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69595042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
Ñ

R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6960352

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
Ç
b
D__inference_dropout_layer_call_and_return_conditional_losses_6965584

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ï

P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966938

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
Ò
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6960080

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
ª
ª
7__inference_batch_normalization_2_layer_call_fn_6967201

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_69599302
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

¯
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6960334

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
ä
ª
7__inference_batch_normalization_1_layer_call_fn_6967121

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_69597262
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
Ò
`
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6967597

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
Ö
`
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6961332

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

¯
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967234

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
¼
ú
I__inference_feed_forward_layer_call_and_return_conditional_losses_6962397

inputs
dense_6962385
dense_6962387
dense_1_6962391
dense_1_6962393
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6962385dense_6962387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_69622632
dense/StatefulPartitionedCallý
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
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
D__inference_dropout_layer_call_and_return_conditional_losses_69622962
dropout/PartitionedCall·
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_6962391dense_1_6962393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_69623192!
dense_1/StatefulPartitionedCall¾
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
ò
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6960756

inputs
conv2d_4_6960740
conv2d_4_6960742!
batch_normalization_4_6960745!
batch_normalization_4_6960747!
batch_normalization_4_6960749!
batch_normalization_4_6960751
identity¢-batch_normalization_4/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCallª
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_6960740conv2d_4_6960742*
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
   E8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_69606122"
 conv2d_4/StatefulPartitionedCallÎ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_6960745batch_normalization_4_6960747batch_normalization_4_6960749batch_normalization_4_6960751*
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
   E8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_69606472/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
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
   E8 *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_69607062
re_lu_4/PartitionedCallÏ
IdentityIdentity re_lu_4/PartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

­
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966920

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
§
­
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6960299

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

¾
-__inference_cnn_block_2_layer_call_fn_6966129

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
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69601662
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
ý
#
"__inference__wrapped_model_6959242
input_1U
Qclass_cnn_feature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resourceV
Rclass_cnn_feature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resource[
Wclass_cnn_feature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resource]
Yclass_cnn_feature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resourcel
hclass_cnn_feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourcen
jclass_cnn_feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceW
Sclass_cnn_feature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resourceX
Tclass_cnn_feature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resource]
Yclass_cnn_feature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resource_
[class_cnn_feature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resourcen
jclass_cnn_feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourcep
lclass_cnn_feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceW
Sclass_cnn_feature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resourceX
Tclass_cnn_feature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resource]
Yclass_cnn_feature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resource_
[class_cnn_feature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resourcen
jclass_cnn_feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourcep
lclass_cnn_feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceW
Sclass_cnn_feature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resourceX
Tclass_cnn_feature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resource]
Yclass_cnn_feature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resource_
[class_cnn_feature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resourcen
jclass_cnn_feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourcep
lclass_cnn_feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceW
Sclass_cnn_feature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resourceX
Tclass_cnn_feature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resource]
Yclass_cnn_feature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resource_
[class_cnn_feature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resourcen
jclass_cnn_feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourcep
lclass_cnn_feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceW
Sclass_cnn_feature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resourceX
Tclass_cnn_feature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resource]
Yclass_cnn_feature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resource_
[class_cnn_feature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resourcen
jclass_cnn_feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourcep
lclass_cnn_feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceW
Sclass_cnn_feature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resourceX
Tclass_cnn_feature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resource]
Yclass_cnn_feature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resource_
[class_cnn_feature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resourcen
jclass_cnn_feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourcep
lclass_cnn_feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource?
;class_cnn_feed_forward_dense_matmul_readvariableop_resource@
<class_cnn_feed_forward_dense_biasadd_readvariableop_resourceA
=class_cnn_feed_forward_dense_1_matmul_readvariableop_resourceB
>class_cnn_feed_forward_dense_1_biasadd_readvariableop_resource
identityo
class_cnn/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
class_cnn/truediv/y
class_cnn/truedivRealDivinput_1class_cnn/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
class_cnn/truediv®
Hclass_cnn/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOpQclass_cnn_feature_extractor_cnn_cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02J
Hclass_cnn/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOpÌ
9class_cnn/feature_extractor_cnn/cnn_block_0/conv2d/Conv2DConv2Dclass_cnn/truediv:z:0Pclass_cnn/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2;
9class_cnn/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D¥
Iclass_cnn/feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOpRclass_cnn_feature_extractor_cnn_cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02K
Iclass_cnn/feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOpÔ
:class_cnn/feature_extractor_cnn/cnn_block_0/conv2d/BiasAddBiasAddBclass_cnn/feature_extractor_cnn/cnn_block_0/conv2d/Conv2D:output:0Qclass_cnn/feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:class_cnn/feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd´
Nclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOpReadVariableOpWclass_cnn_feature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02P
Nclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOpº
Pclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOpYclass_cnn_feature_extractor_cnn_cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Pclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1ç
_class_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOphclass_cnn_feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02a
_class_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpí
aclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjclass_cnn_feature_extractor_cnn_cnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02c
aclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1
Pclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV3Cclass_cnn/feature_extractor_cnn/cnn_block_0/conv2d/BiasAdd:output:0Vclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp:value:0Xclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/ReadVariableOp_1:value:0gclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0iclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2R
Pclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3
6class_cnn/feature_extractor_cnn/cnn_block_0/re_lu/ReluReluTclass_cnn/feature_extractor_cnn/cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6class_cnn/feature_extractor_cnn/cnn_block_0/re_lu/Relu´
Jclass_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpSclass_cnn_feature_extractor_cnn_cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02L
Jclass_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOp
;class_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2DConv2DDclass_cnn/feature_extractor_cnn/cnn_block_0/re_lu/Relu:activations:0Rclass_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2=
;class_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D«
Kclass_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpTclass_cnn_feature_extractor_cnn_cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02M
Kclass_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpÜ
<class_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAddBiasAddDclass_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/Conv2D:output:0Sclass_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2>
<class_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAddº
Pclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOpYclass_cnn_feature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02R
Pclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOpÀ
Rclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp[class_cnn_feature_extractor_cnn_cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02T
Rclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1í
aclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpjclass_cnn_feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02c
aclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpó
cclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplclass_cnn_feature_extractor_cnn_cnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02e
cclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1
Rclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3Eclass_cnn/feature_extractor_cnn/cnn_block_1/conv2d_1/BiasAdd:output:0Xclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp:value:0Zclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0iclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0kclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2T
Rclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3
8class_cnn/feature_extractor_cnn/cnn_block_1/re_lu_1/ReluReluVclass_cnn/feature_extractor_cnn/cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2:
8class_cnn/feature_extractor_cnn/cnn_block_1/re_lu_1/Relu´
Jclass_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpSclass_cnn_feature_extractor_cnn_cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02L
Jclass_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOp
;class_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2DConv2DFclass_cnn/feature_extractor_cnn/cnn_block_1/re_lu_1/Relu:activations:0Rclass_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2=
;class_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D«
Kclass_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpTclass_cnn_feature_extractor_cnn_cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02M
Kclass_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpÜ
<class_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAddBiasAddDclass_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/Conv2D:output:0Sclass_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2>
<class_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAddº
Pclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOpYclass_cnn_feature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02R
Pclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOpÀ
Rclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp[class_cnn_feature_extractor_cnn_cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02T
Rclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1í
aclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpjclass_cnn_feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02c
aclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpó
cclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplclass_cnn_feature_extractor_cnn_cnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02e
cclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1
Rclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Eclass_cnn/feature_extractor_cnn/cnn_block_2/conv2d_2/BiasAdd:output:0Xclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp:value:0Zclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0iclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0kclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2T
Rclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3
8class_cnn/feature_extractor_cnn/cnn_block_2/re_lu_2/ReluReluVclass_cnn/feature_extractor_cnn/cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2:
8class_cnn/feature_extractor_cnn/cnn_block_2/re_lu_2/Relu´
Jclass_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpSclass_cnn_feature_extractor_cnn_cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02L
Jclass_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp
;class_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2DConv2DFclass_cnn/feature_extractor_cnn/cnn_block_2/re_lu_2/Relu:activations:0Rclass_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2=
;class_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D«
Kclass_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpTclass_cnn_feature_extractor_cnn_cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02M
Kclass_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpÜ
<class_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAddBiasAddDclass_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/Conv2D:output:0Sclass_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2>
<class_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAddº
Pclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOpYclass_cnn_feature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02R
Pclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOpÀ
Rclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp[class_cnn_feature_extractor_cnn_cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02T
Rclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1í
aclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpjclass_cnn_feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02c
aclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpó
cclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplclass_cnn_feature_extractor_cnn_cnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02e
cclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1
Rclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3Eclass_cnn/feature_extractor_cnn/cnn_block_3/conv2d_3/BiasAdd:output:0Xclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp:value:0Zclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0iclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0kclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2T
Rclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3
8class_cnn/feature_extractor_cnn/cnn_block_3/re_lu_3/ReluReluVclass_cnn/feature_extractor_cnn/cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2:
8class_cnn/feature_extractor_cnn/cnn_block_3/re_lu_3/Relu´
Jclass_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpSclass_cnn_feature_extractor_cnn_cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02L
Jclass_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp
;class_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2DConv2DFclass_cnn/feature_extractor_cnn/cnn_block_3/re_lu_3/Relu:activations:0Rclass_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2=
;class_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D«
Kclass_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpTclass_cnn_feature_extractor_cnn_cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02M
Kclass_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpÜ
<class_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAddBiasAddDclass_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/Conv2D:output:0Sclass_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2>
<class_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAddº
Pclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOpYclass_cnn_feature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02R
Pclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOpÀ
Rclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp[class_cnn_feature_extractor_cnn_cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02T
Rclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1í
aclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpjclass_cnn_feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02c
aclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpó
cclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplclass_cnn_feature_extractor_cnn_cnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02e
cclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1
Rclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3Eclass_cnn/feature_extractor_cnn/cnn_block_4/conv2d_4/BiasAdd:output:0Xclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp:value:0Zclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0iclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0kclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2T
Rclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3
8class_cnn/feature_extractor_cnn/cnn_block_4/re_lu_4/ReluReluVclass_cnn/feature_extractor_cnn/cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2:
8class_cnn/feature_extractor_cnn/cnn_block_4/re_lu_4/Relu´
Jclass_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpSclass_cnn_feature_extractor_cnn_cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02L
Jclass_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp
;class_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2DConv2DFclass_cnn/feature_extractor_cnn/cnn_block_4/re_lu_4/Relu:activations:0Rclass_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2=
;class_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D«
Kclass_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpTclass_cnn_feature_extractor_cnn_cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02M
Kclass_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpÜ
<class_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAddBiasAddDclass_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/Conv2D:output:0Sclass_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2>
<class_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAddº
Pclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOpYclass_cnn_feature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02R
Pclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOpÀ
Rclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp[class_cnn_feature_extractor_cnn_cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02T
Rclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1í
aclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpjclass_cnn_feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02c
aclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpó
cclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplclass_cnn_feature_extractor_cnn_cnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02e
cclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1
Rclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3Eclass_cnn/feature_extractor_cnn/cnn_block_5/conv2d_5/BiasAdd:output:0Xclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp:value:0Zclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0iclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0kclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2T
Rclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3
8class_cnn/feature_extractor_cnn/cnn_block_5/re_lu_5/ReluReluVclass_cnn/feature_extractor_cnn/cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2:
8class_cnn/feature_extractor_cnn/cnn_block_5/re_lu_5/Reluµ
Jclass_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpSclass_cnn_feature_extractor_cnn_cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02L
Jclass_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOp
;class_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2DConv2DFclass_cnn/feature_extractor_cnn/cnn_block_5/re_lu_5/Relu:activations:0Rclass_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2=
;class_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D¬
Kclass_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpTclass_cnn_feature_extractor_cnn_cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02M
Kclass_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpÝ
<class_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAddBiasAddDclass_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/Conv2D:output:0Sclass_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<class_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd»
Pclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOpYclass_cnn_feature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02R
Pclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOpÁ
Rclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp[class_cnn_feature_extractor_cnn_cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02T
Rclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1î
aclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpjclass_cnn_feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02c
aclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpô
cclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplclass_cnn_feature_extractor_cnn_cnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02e
cclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1
Rclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3Eclass_cnn/feature_extractor_cnn/cnn_block_6/conv2d_6/BiasAdd:output:0Xclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp:value:0Zclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0iclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0kclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2T
Rclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3
8class_cnn/feature_extractor_cnn/cnn_block_6/re_lu_6/ReluReluVclass_cnn/feature_extractor_cnn/cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8class_cnn/feature_extractor_cnn/cnn_block_6/re_lu_6/Reluó
Oclass_cnn/feature_extractor_cnn/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2Q
Oclass_cnn/feature_extractor_cnn/global_average_pooling2d/Mean/reduction_indicesÛ
=class_cnn/feature_extractor_cnn/global_average_pooling2d/MeanMeanFclass_cnn/feature_extractor_cnn/cnn_block_6/re_lu_6/Relu:activations:0Xclass_cnn/feature_extractor_cnn/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=class_cnn/feature_extractor_cnn/global_average_pooling2d/Meanå
2class_cnn/feed_forward/dense/MatMul/ReadVariableOpReadVariableOp;class_cnn_feed_forward_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype024
2class_cnn/feed_forward/dense/MatMul/ReadVariableOp
#class_cnn/feed_forward/dense/MatMulMatMulFclass_cnn/feature_extractor_cnn/global_average_pooling2d/Mean:output:0:class_cnn/feed_forward/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#class_cnn/feed_forward/dense/MatMulã
3class_cnn/feed_forward/dense/BiasAdd/ReadVariableOpReadVariableOp<class_cnn_feed_forward_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3class_cnn/feed_forward/dense/BiasAdd/ReadVariableOpõ
$class_cnn/feed_forward/dense/BiasAddBiasAdd-class_cnn/feed_forward/dense/MatMul:product:0;class_cnn/feed_forward/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$class_cnn/feed_forward/dense/BiasAdd¯
!class_cnn/feed_forward/dense/ReluRelu-class_cnn/feed_forward/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!class_cnn/feed_forward/dense/ReluÁ
'class_cnn/feed_forward/dropout/IdentityIdentity/class_cnn/feed_forward/dense/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'class_cnn/feed_forward/dropout/Identityê
4class_cnn/feed_forward/dense_1/MatMul/ReadVariableOpReadVariableOp=class_cnn_feed_forward_dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype026
4class_cnn/feed_forward/dense_1/MatMul/ReadVariableOpú
%class_cnn/feed_forward/dense_1/MatMulMatMul0class_cnn/feed_forward/dropout/Identity:output:0<class_cnn/feed_forward/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%class_cnn/feed_forward/dense_1/MatMulé
5class_cnn/feed_forward/dense_1/BiasAdd/ReadVariableOpReadVariableOp>class_cnn_feed_forward_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype027
5class_cnn/feed_forward/dense_1/BiasAdd/ReadVariableOpý
&class_cnn/feed_forward/dense_1/BiasAddBiasAdd/class_cnn/feed_forward/dense_1/MatMul:product:0=class_cnn/feed_forward/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&class_cnn/feed_forward/dense_1/BiasAdd
IdentityIdentity/class_cnn/feed_forward/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
°
¨
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965665

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identityª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ó
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

re_lu/Relut
IdentityIdentityre_lu/Relu:activations:0*
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
Ñ

R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6959726

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
¨
Æ
-__inference_cnn_block_4_layer_call_fn_6966473
conv2d_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69607922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_4_input
¦
Æ
-__inference_cnn_block_2_layer_call_fn_6966026
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_69601302
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
_user_specified_nameconv2d_2_input
Ý

R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6961291

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
Ò
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6967126

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

¯
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6959708

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
-__inference_cnn_block_3_layer_call_fn_6966301

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
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69604792
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

¯
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6960021

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
Ö
¯
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6961182

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
°
ª
7__inference_batch_normalization_6_layer_call_fn_6967842

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
   E8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69612132
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
Ã
¦
.__inference_feed_forward_layer_call_fn_6965465
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *R
fMRK
I__inference_feed_forward_layer_call_and_return_conditional_losses_69623692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input

¼
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966525
conv2d_5_input+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpÇ
conv2d_5/Conv2DConv2Dconv2d_5_input&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_5/Conv2D§
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp¬
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_5/BiasAdd¶
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp¼
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1é
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_5/Reluv
IdentityIdentityre_lu_5/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@:::::::_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_5_input
Þ
¨
5__inference_batch_normalization_layer_call_fn_6966951

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
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
   E8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_69593952
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


*__inference_conv2d_2_layer_call_fn_6967150

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
E__inference_conv2d_2_layer_call_and_return_conditional_losses_69599862
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
Ñ

R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967095

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
Ð
^
B__inference_re_lu_layer_call_and_return_conditional_losses_6959454

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
¦
Æ
-__inference_cnn_block_4_layer_call_fn_6966456
conv2d_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_69607562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_4_input

´
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966095

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_2/Conv2D/ReadVariableOp¿
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_2/BiasAdd¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_2/Reluv
IdentityIdentityre_lu_2/Relu:activations:0*
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
´
¡
.__inference_feed_forward_layer_call_fn_6965547

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *R
fMRK
I__inference_feed_forward_layer_call_and_return_conditional_losses_69623972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
ò
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6961069

inputs
conv2d_5_6961053
conv2d_5_6961055!
batch_normalization_5_6961058!
batch_normalization_5_6961060!
batch_normalization_5_6961062!
batch_normalization_5_6961064
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCallª
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_6961053conv2d_5_6961055*
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
   E8 *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_69609252"
 conv2d_5/StatefulPartitionedCallÎ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_6961058batch_normalization_5_6961060batch_normalization_5_6961062batch_normalization_5_6961064*
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
   E8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_69609602/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
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
   E8 *M
fHRF
D__inference_re_lu_5_layer_call_and_return_conditional_losses_69610192
re_lu_5/PartitionedCallÏ
IdentityIdentity re_lu_5/PartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
¬
D__inference_dense_1_layer_call_and_return_conditional_losses_6965604

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§
­
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6967298

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
Ï

P__inference_batch_normalization_layer_call_and_return_conditional_losses_6959413

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


R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967031

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
¦
Æ
-__inference_cnn_block_3_layer_call_fn_6966198
conv2d_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69604432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_3_input
í

H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966328

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1°
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_4/Conv2D/ReadVariableOp¿
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_4/Conv2D§
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp¬
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_4/BiasAdd¶
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp¼
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1é
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_4/FusedBatchNormV3
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_4/ReluÆ
IdentityIdentityre_lu_4/Relu:activations:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

}
(__inference_conv2d_layer_call_fn_6966836

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
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
   E8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_69593602
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

¯
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967391

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
Ñ

R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967409

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
÷­
ù
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6965231
input_15
1cnn_block_0_conv2d_conv2d_readvariableop_resource6
2cnn_block_0_conv2d_biasadd_readvariableop_resource;
7cnn_block_0_batch_normalization_readvariableop_resource=
9cnn_block_0_batch_normalization_readvariableop_1_resourceL
Hcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resourceN
Jcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_1_conv2d_1_conv2d_readvariableop_resource8
4cnn_block_1_conv2d_1_biasadd_readvariableop_resource=
9cnn_block_1_batch_normalization_1_readvariableop_resource?
;cnn_block_1_batch_normalization_1_readvariableop_1_resourceN
Jcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_2_conv2d_2_conv2d_readvariableop_resource8
4cnn_block_2_conv2d_2_biasadd_readvariableop_resource=
9cnn_block_2_batch_normalization_2_readvariableop_resource?
;cnn_block_2_batch_normalization_2_readvariableop_1_resourceN
Jcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_3_conv2d_3_conv2d_readvariableop_resource8
4cnn_block_3_conv2d_3_biasadd_readvariableop_resource=
9cnn_block_3_batch_normalization_3_readvariableop_resource?
;cnn_block_3_batch_normalization_3_readvariableop_1_resourceN
Jcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_4_conv2d_4_conv2d_readvariableop_resource8
4cnn_block_4_conv2d_4_biasadd_readvariableop_resource=
9cnn_block_4_batch_normalization_4_readvariableop_resource?
;cnn_block_4_batch_normalization_4_readvariableop_1_resourceN
Jcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_5_conv2d_5_conv2d_readvariableop_resource8
4cnn_block_5_conv2d_5_biasadd_readvariableop_resource=
9cnn_block_5_batch_normalization_5_readvariableop_resource?
;cnn_block_5_batch_normalization_5_readvariableop_1_resourceN
Jcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3cnn_block_6_conv2d_6_conv2d_readvariableop_resource8
4cnn_block_6_conv2d_6_biasadd_readvariableop_resource=
9cnn_block_6_batch_normalization_6_readvariableop_resource?
;cnn_block_6_batch_normalization_6_readvariableop_1_resourceN
Jcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identityÎ
(cnn_block_0/conv2d/Conv2D/ReadVariableOpReadVariableOp1cnn_block_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(cnn_block_0/conv2d/Conv2D/ReadVariableOpÞ
cnn_block_0/conv2d/Conv2DConv2Dinput_10cnn_block_0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_0/conv2d/Conv2DÅ
)cnn_block_0/conv2d/BiasAdd/ReadVariableOpReadVariableOp2cnn_block_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)cnn_block_0/conv2d/BiasAdd/ReadVariableOpÔ
cnn_block_0/conv2d/BiasAddBiasAdd"cnn_block_0/conv2d/Conv2D:output:01cnn_block_0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/conv2d/BiasAddÔ
.cnn_block_0/batch_normalization/ReadVariableOpReadVariableOp7cnn_block_0_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype020
.cnn_block_0/batch_normalization/ReadVariableOpÚ
0cnn_block_0/batch_normalization/ReadVariableOp_1ReadVariableOp9cnn_block_0_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype022
0cnn_block_0/batch_normalization/ReadVariableOp_1
?cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?cnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp
Acnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJcnn_block_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Acnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1§
0cnn_block_0/batch_normalization/FusedBatchNormV3FusedBatchNormV3#cnn_block_0/conv2d/BiasAdd:output:06cnn_block_0/batch_normalization/ReadVariableOp:value:08cnn_block_0/batch_normalization/ReadVariableOp_1:value:0Gcnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Icnn_block_0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 22
0cnn_block_0/batch_normalization/FusedBatchNormV3¨
cnn_block_0/re_lu/ReluRelu4cnn_block_0/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_0/re_lu/ReluÔ
*cnn_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3cnn_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_block_1/conv2d_1/Conv2D/ReadVariableOp
cnn_block_1/conv2d_1/Conv2DConv2D$cnn_block_0/re_lu/Relu:activations:02cnn_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_1/conv2d_1/Conv2DË
+cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_1/conv2d_1/BiasAdd/ReadVariableOpÜ
cnn_block_1/conv2d_1/BiasAddBiasAdd$cnn_block_1/conv2d_1/Conv2D:output:03cnn_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/conv2d_1/BiasAddÚ
0cnn_block_1/batch_normalization_1/ReadVariableOpReadVariableOp9cnn_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_1/batch_normalization_1/ReadVariableOpà
2cnn_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp;cnn_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_1/batch_normalization_1/ReadVariableOp_1
Acnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Ccnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%cnn_block_1/conv2d_1/BiasAdd:output:08cnn_block_1/batch_normalization_1/ReadVariableOp:value:0:cnn_block_1/batch_normalization_1/ReadVariableOp_1:value:0Icnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_1/batch_normalization_1/FusedBatchNormV3®
cnn_block_1/re_lu_1/ReluRelu6cnn_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_1/re_lu_1/ReluÔ
*cnn_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3cnn_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_2/conv2d_2/Conv2D/ReadVariableOp
cnn_block_2/conv2d_2/Conv2DConv2D&cnn_block_1/re_lu_1/Relu:activations:02cnn_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_2/conv2d_2/Conv2DË
+cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_2/conv2d_2/BiasAdd/ReadVariableOpÜ
cnn_block_2/conv2d_2/BiasAddBiasAdd$cnn_block_2/conv2d_2/Conv2D:output:03cnn_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/conv2d_2/BiasAddÚ
0cnn_block_2/batch_normalization_2/ReadVariableOpReadVariableOp9cnn_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_2/batch_normalization_2/ReadVariableOpà
2cnn_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp;cnn_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_2/batch_normalization_2/ReadVariableOp_1
Acnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Ccnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%cnn_block_2/conv2d_2/BiasAdd:output:08cnn_block_2/batch_normalization_2/ReadVariableOp:value:0:cnn_block_2/batch_normalization_2/ReadVariableOp_1:value:0Icnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_2/batch_normalization_2/FusedBatchNormV3®
cnn_block_2/re_lu_2/ReluRelu6cnn_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_2/re_lu_2/ReluÔ
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3cnn_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*cnn_block_3/conv2d_3/Conv2D/ReadVariableOp
cnn_block_3/conv2d_3/Conv2DConv2D&cnn_block_2/re_lu_2/Relu:activations:02cnn_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
cnn_block_3/conv2d_3/Conv2DË
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_block_3/conv2d_3/BiasAdd/ReadVariableOpÜ
cnn_block_3/conv2d_3/BiasAddBiasAdd$cnn_block_3/conv2d_3/Conv2D:output:03cnn_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/conv2d_3/BiasAddÚ
0cnn_block_3/batch_normalization_3/ReadVariableOpReadVariableOp9cnn_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype022
0cnn_block_3/batch_normalization_3/ReadVariableOpà
2cnn_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp;cnn_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2cnn_block_3/batch_normalization_3/ReadVariableOp_1
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Acnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Ccnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%cnn_block_3/conv2d_3/BiasAdd:output:08cnn_block_3/batch_normalization_3/ReadVariableOp:value:0:cnn_block_3/batch_normalization_3/ReadVariableOp_1:value:0Icnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2cnn_block_3/batch_normalization_3/FusedBatchNormV3®
cnn_block_3/re_lu_3/ReluRelu6cnn_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cnn_block_3/re_lu_3/ReluÔ
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3cnn_block_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_block_4/conv2d_4/Conv2D/ReadVariableOp
cnn_block_4/conv2d_4/Conv2DConv2D&cnn_block_3/re_lu_3/Relu:activations:02cnn_block_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_4/conv2d_4/Conv2DË
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_4/conv2d_4/BiasAdd/ReadVariableOpÜ
cnn_block_4/conv2d_4/BiasAddBiasAdd$cnn_block_4/conv2d_4/Conv2D:output:03cnn_block_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/conv2d_4/BiasAddÚ
0cnn_block_4/batch_normalization_4/ReadVariableOpReadVariableOp9cnn_block_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_4/batch_normalization_4/ReadVariableOpà
2cnn_block_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp;cnn_block_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_4/batch_normalization_4/ReadVariableOp_1
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%cnn_block_4/conv2d_4/BiasAdd:output:08cnn_block_4/batch_normalization_4/ReadVariableOp:value:0:cnn_block_4/batch_normalization_4/ReadVariableOp_1:value:0Icnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 24
2cnn_block_4/batch_normalization_4/FusedBatchNormV3®
cnn_block_4/re_lu_4/ReluRelu6cnn_block_4/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_4/re_lu_4/ReluÔ
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3cnn_block_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*cnn_block_5/conv2d_5/Conv2D/ReadVariableOp
cnn_block_5/conv2d_5/Conv2DConv2D&cnn_block_4/re_lu_4/Relu:activations:02cnn_block_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
cnn_block_5/conv2d_5/Conv2DË
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_block_5/conv2d_5/BiasAdd/ReadVariableOpÜ
cnn_block_5/conv2d_5/BiasAddBiasAdd$cnn_block_5/conv2d_5/Conv2D:output:03cnn_block_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/conv2d_5/BiasAddÚ
0cnn_block_5/batch_normalization_5/ReadVariableOpReadVariableOp9cnn_block_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0cnn_block_5/batch_normalization_5/ReadVariableOpà
2cnn_block_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp;cnn_block_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2cnn_block_5/batch_normalization_5/ReadVariableOp_1
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Acnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Ccnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1µ
2cnn_block_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%cnn_block_5/conv2d_5/BiasAdd:output:08cnn_block_5/batch_normalization_5/ReadVariableOp:value:0:cnn_block_5/batch_normalization_5/ReadVariableOp_1:value:0Icnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 24
2cnn_block_5/batch_normalization_5/FusedBatchNormV3®
cnn_block_5/re_lu_5/ReluRelu6cnn_block_5/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
cnn_block_5/re_lu_5/ReluÕ
*cnn_block_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3cnn_block_6_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*cnn_block_6/conv2d_6/Conv2D/ReadVariableOp
cnn_block_6/conv2d_6/Conv2DConv2D&cnn_block_5/re_lu_5/Relu:activations:02cnn_block_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
cnn_block_6/conv2d_6/Conv2DÌ
+cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4cnn_block_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+cnn_block_6/conv2d_6/BiasAdd/ReadVariableOpÝ
cnn_block_6/conv2d_6/BiasAddBiasAdd$cnn_block_6/conv2d_6/Conv2D:output:03cnn_block_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/conv2d_6/BiasAddÛ
0cnn_block_6/batch_normalization_6/ReadVariableOpReadVariableOp9cnn_block_6_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype022
0cnn_block_6/batch_normalization_6/ReadVariableOpá
2cnn_block_6/batch_normalization_6/ReadVariableOp_1ReadVariableOp;cnn_block_6_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2cnn_block_6/batch_normalization_6/ReadVariableOp_1
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
Acnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLcnn_block_6_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
Ccnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1º
2cnn_block_6/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%cnn_block_6/conv2d_6/BiasAdd:output:08cnn_block_6/batch_normalization_6/ReadVariableOp:value:0:cnn_block_6/batch_normalization_6/ReadVariableOp_1:value:0Icnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kcnn_block_6/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 24
2cnn_block_6/batch_normalization_6/FusedBatchNormV3¯
cnn_block_6/re_lu_6/ReluRelu6cnn_block_6/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cnn_block_6/re_lu_6/Relu³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesÛ
global_average_pooling2d/MeanMean&cnn_block_6/re_lu_6/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_average_pooling2d/Mean{
IdentityIdentity&global_average_pooling2d/Mean:output:0*
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
ä
ª
7__inference_batch_normalization_4_layer_call_fn_6967528

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
:ÿÿÿÿÿÿÿÿÿ@*&
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_69606652
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
¨
Æ
-__inference_cnn_block_6_layer_call_fn_6966714
conv2d_6_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69613822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_6_input

¯
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967862

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
¨
Æ
-__inference_cnn_block_5_layer_call_fn_6966559
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_69611052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_5_input
â
ª
7__inference_batch_normalization_2_layer_call_fn_6967265

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_69600212
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
Ñ

R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6960978

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


R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6960587

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


*__inference_conv2d_6_layer_call_fn_6967778

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
   E8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_69612382
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


R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6960900

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
¬
ª
7__inference_batch_normalization_3_layer_call_fn_6967371

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_69602742
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
Å
·
+__inference_class_cnn_layer_call_fn_6964413

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

unknown_41

unknown_42

unknown_43

unknown_44
identity¢StatefulPartitionedCallà
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
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*8
config_proto(&

CPU

GPU2*0J

   E8 *O
fJRH
F__inference_class_cnn_layer_call_and_return_conditional_losses_69631132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965837
conv2d_1_input+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÇ
conv2d_1/Conv2DConv2Dconv2d_1_input&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_1/BiasAdd¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_1/Reluv
IdentityIdentityre_lu_1/Relu:activations:0*
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
_user_specified_nameconv2d_1_input
 

H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965812
conv2d_1_input+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÇ
conv2d_1/Conv2DConv2Dconv2d_1_input&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_1/BiasAdd¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_1/FusedBatchNormV3
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_1/ReluÆ
IdentityIdentityre_lu_1/Relu:activations:0%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameconv2d_1_input
í

H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966070

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_2/Conv2D/ReadVariableOp¿
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_2/BiasAdd¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_2/FusedBatchNormV3
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_2/ReluÆ
IdentityIdentityre_lu_2/Relu:activations:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6959617

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


R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6959961

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
Ê
¯
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967013

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

­
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6959395

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
Ò
`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6959767

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
¨
Æ
-__inference_cnn_block_3_layer_call_fn_6966215
conv2d_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_69604792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_3_input
½
E
)__inference_re_lu_5_layer_call_fn_6967759

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
:ÿÿÿÿÿÿÿÿÿ@* 
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
D__inference_re_lu_5_layer_call_and_return_conditional_losses_69610192
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

´
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966783

inputs+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity±
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpÀ
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_6/Conv2D¨
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_6/BiasAdd·
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_6/ReadVariableOp½
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1ê
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1æ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_6/Reluw
IdentityIdentityre_lu_6/Relu:activations:0*
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
¢
Ä
-__inference_cnn_block_0_layer_call_fn_6965785
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69595402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
í

H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965898

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp¿
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_1/BiasAdd¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_1/FusedBatchNormV3
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_1/ReluÆ
IdentityIdentityre_lu_1/Relu:activations:0%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
E
)__inference_re_lu_4_layer_call_fn_6967602

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
:ÿÿÿÿÿÿÿÿÿ@* 
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
D__inference_re_lu_4_layer_call_and_return_conditional_losses_69607062
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

¾
-__inference_cnn_block_0_layer_call_fn_6965682

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
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69595042
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
¦
¨
5__inference_batch_normalization_layer_call_fn_6966887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall½
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
   E8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_69593042
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
Å
ò
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6960479

inputs
conv2d_3_6960463
conv2d_3_6960465!
batch_normalization_3_6960468!
batch_normalization_3_6960470!
batch_normalization_3_6960472!
batch_normalization_3_6960474
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCallª
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_6960463conv2d_3_6960465*
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
   E8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_69602992"
 conv2d_3/StatefulPartitionedCallÐ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_6960468batch_normalization_3_6960470batch_normalization_3_6960472batch_normalization_3_6960474*
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
   E8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_69603522/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
   E8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_69603932
re_lu_3/PartitionedCallÏ
IdentityIdentity re_lu_3/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967566

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
 

H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966414
conv2d_4_input+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1°
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_4/Conv2D/ReadVariableOpÇ
conv2d_4/Conv2DConv2Dconv2d_4_input&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_4/Conv2D§
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp¬
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_4/BiasAdd¶
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp¼
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1é
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_4/FusedBatchNormV3
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_4/ReluÆ
IdentityIdentityre_lu_4/Relu:activations:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_1:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_4_input

c
D__inference_dropout_layer_call_and_return_conditional_losses_6965579

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç
ò
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6961382

inputs
conv2d_6_6961366
conv2d_6_6961368!
batch_normalization_6_6961371!
batch_normalization_6_6961373!
batch_normalization_6_6961375!
batch_normalization_6_6961377
identity¢-batch_normalization_6/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall«
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_6961366conv2d_6_6961368*
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
   E8 *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_69612382"
 conv2d_6/StatefulPartitionedCallÏ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_6961371batch_normalization_6_6961373batch_normalization_6_6961375batch_normalization_6_6961377*
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
   E8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69612732/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
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
   E8 *M
fHRF
D__inference_re_lu_6_layer_call_and_return_conditional_losses_69613322
re_lu_6/PartitionedCallÐ
IdentityIdentity re_lu_6/PartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

´
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966611

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp¿
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_5/Conv2D§
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp¬
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_5/BiasAdd¶
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp¼
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1é
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_5/Reluv
IdentityIdentityre_lu_5/Relu:activations:0*
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

¼
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966181
conv2d_3_input+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOpÇ
conv2d_3/Conv2DConv2Dconv2d_3_input&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp¬
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_3/BiasAdd¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1á
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
re_lu_3/Reluv
IdentityIdentityre_lu_3/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ :::::::_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_3_input

¾
-__inference_cnn_block_0_layer_call_fn_6965699

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
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_69595402
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
¦

R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6961213

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
Ñ

R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6960039

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


R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967659

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
Ã
¦
.__inference_feed_forward_layer_call_fn_6965478
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

   E8 *R
fMRK
I__inference_feed_forward_layer_call_and_return_conditional_losses_69623972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input


R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6960274

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
í

H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966586

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_5/AssignNewValue¢&batch_normalization_5/AssignNewValue_1°
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp¿
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_5/Conv2D§
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp¬
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_5/BiasAdd¶
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp¼
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1é
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ï
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_5/FusedBatchNormV3
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
re_lu_5/ReluÆ
IdentityIdentityre_lu_5/Relu:activations:0%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@::::::2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¥
«
C__inference_conv2d_layer_call_and_return_conditional_losses_6966827

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
Ê
¯
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967327

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
¬
ª
7__inference_batch_normalization_5_layer_call_fn_6967685

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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_69609002
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
Ý

R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967880

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
è
ª
7__inference_batch_normalization_6_layer_call_fn_6967906

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
   E8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69612912
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

²
%__inference_signature_wrapper_6963307
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

unknown_41

unknown_42

unknown_43

unknown_44
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
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*8
config_proto(&

CPU

GPU2*0J

   E8 *+
f&R$
"__inference__wrapped_model_69592422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*è
_input_shapesÖ
Ó:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
½
E
)__inference_re_lu_3_layer_call_fn_6967445

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
:ÿÿÿÿÿÿÿÿÿ * 
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
D__inference_re_lu_3_layer_call_and_return_conditional_losses_69603932
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
¦

I__inference_feed_forward_layer_call_and_return_conditional_losses_6965452
dense_input(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense_input#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Identity¥
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::::U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
Ð
^
B__inference_re_lu_layer_call_and_return_conditional_losses_6966969

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


P__inference_batch_normalization_layer_call_and_return_conditional_losses_6959335

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
Å
ò
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6960792

inputs
conv2d_4_6960776
conv2d_4_6960778!
batch_normalization_4_6960781!
batch_normalization_4_6960783!
batch_normalization_4_6960785!
batch_normalization_4_6960787
identity¢-batch_normalization_4/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCallª
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_6960776conv2d_4_6960778*
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
   E8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_69606122"
 conv2d_4/StatefulPartitionedCallÐ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_6960781batch_normalization_4_6960783batch_normalization_4_6960785batch_normalization_4_6960787*
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
   E8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_69606652/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
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
   E8 *M
fHRF
D__inference_re_lu_4_layer_call_and_return_conditional_losses_69607062
re_lu_4/PartitionedCallÏ
IdentityIdentity re_lu_4/PartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

â
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6959540

inputs
conv2d_6959524
conv2d_6959526
batch_normalization_6959529
batch_normalization_6959531
batch_normalization_6959533
batch_normalization_6959535
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall 
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6959524conv2d_6959526*
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
   E8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_69593602 
conv2d/StatefulPartitionedCallÀ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_6959529batch_normalization_6959531batch_normalization_6959533batch_normalization_6959535*
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
   E8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_69594132-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
   E8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_69594542
re_lu/PartitionedCallÉ
IdentityIdentityre_lu/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
`
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6960706

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
Ê
¯
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6959930

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
©
b
)__inference_dropout_layer_call_fn_6965589

inputs
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
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
D__inference_dropout_layer_call_and_return_conditional_losses_69622912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
É

7__inference_feature_extractor_cnn_layer_call_fn_6964822

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
identity¢StatefulPartitionedCall§
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
   E8 *[
fVRT
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_69619762
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
×

7__inference_feature_extractor_cnn_layer_call_fn_6964911

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
identity¢StatefulPartitionedCallµ
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
   E8 *[
fVRT
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_69621612
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
«
¼
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966697
conv2d_6_input+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource
identity±
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOpÈ
conv2d_6/Conv2DConv2Dconv2d_6_input&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_6/Conv2D¨
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_6/BiasAdd·
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_6/ReadVariableOp½
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1ê
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1æ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_6/Reluw
IdentityIdentityre_lu_6/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ@:::::::_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_6_input

¾
-__inference_cnn_block_6_layer_call_fn_6966800

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
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_69613822
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
¬
ª
7__inference_batch_normalization_4_layer_call_fn_6967592

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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_69605872
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:£Ö
Ú
feature_extractor
head
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+½&call_and_return_all_conditional_losses
¾__call__
¿_default_save_signature"ú
_tf_keras_modelà{"class_name": "ClassCNN", "name": "class_cnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ClassCNN"}}
Æ

cnn_blocks
	gba

	variables
regularization_losses
trainable_variables
	keras_api
+À&call_and_return_all_conditional_losses
Á__call__"
_tf_keras_model{"class_name": "FeatureExtractorCNN", "name": "feature_extractor_cnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "FeatureExtractorCNN"}}
À
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
+Â&call_and_return_all_conditional_losses
Ã__call__"Ô
_tf_keras_sequentialµ{"class_name": "Sequential", "name": "feed_forward", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "feed_forward", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "feed_forward", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}

0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
)20
*21
+22
,23
-24
.25
/26
027
128
229
330
431
532
633
734
835
936
:37
;38
<39
=40
>41
?42
@43
A44
B45"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
)20
*21
+22
,23
-24
.25
/26
027
?28
@29
A30
B31"
trackable_list_wrapper
Î
	variables
Clayer_metrics

Dlayers
Enon_trainable_variables
regularization_losses
Fmetrics
Glayer_regularization_losses
trainable_variables
¾__call__
¿_default_save_signature
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
-
Äserving_default"
signature_map
Q
H0
I1
J2
K3
L4
M5
N6"
trackable_list_wrapper

O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
+Å&call_and_return_all_conditional_losses
Æ__call__"
_tf_keras_layerê{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
æ
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
)20
*21
+22
,23
-24
.25
/26
027
128
229
330
431
532
633
734
835
936
:37
;38
<39
=40
>41"
trackable_list_wrapper
 "
trackable_list_wrapper
ö
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
)20
*21
+22
,23
-24
.25
/26
027"
trackable_list_wrapper
°

	variables
Slayer_metrics

Tlayers
Unon_trainable_variables
regularization_losses
Vmetrics
Wlayer_regularization_losses
trainable_variables
Á__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object

X_inbound_nodes

?kernel
@bias
Y_outbound_nodes
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+Ç&call_and_return_all_conditional_losses
È__call__"Æ
_tf_keras_layer¬{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}

^_inbound_nodes
__outbound_nodes
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+É&call_and_return_all_conditional_losses
Ê__call__"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}

d_inbound_nodes

Akernel
Bbias
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
+Ë&call_and_return_all_conditional_losses
Ì__call__"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 64]}}
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
°
	variables
ilayer_metrics

jlayers
knon_trainable_variables
regularization_losses
lmetrics
mlayer_regularization_losses
trainable_variables
Ã__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
':%2batch_normalization/gamma
&:$2batch_normalization/beta
):' 2conv2d_1/kernel
: 2conv2d_1/bias
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
):'  2conv2d_2/kernel
: 2conv2d_2/bias
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
):'  2conv2d_3/kernel
: 2conv2d_3/bias
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
):' @2conv2d_4/kernel
:@2conv2d_4/bias
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
*:(@2conv2d_6/kernel
:2conv2d_6/bias
*:(2batch_normalization_6/gamma
):'2batch_normalization_6/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
2:0 (2!batch_normalization_6/moving_mean
6:4 (2%batch_normalization_6/moving_variance
:	@2dense/kernel
:@2
dense/bias
 :@
2dense_1/kernel
:
2dense_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper

10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Û
nlayer_with_weights-0
nlayer-0
olayer_with_weights-1
olayer-1
player-2
q	variables
rregularization_losses
strainable_variables
t	keras_api
+Í&call_and_return_all_conditional_losses
Î__call__"ï
_tf_keras_sequentialÐ{"class_name": "Sequential", "name": "cnn_block_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ï
ulayer_with_weights-0
ulayer-0
vlayer_with_weights-1
vlayer-1
wlayer-2
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
+Ï&call_and_return_all_conditional_losses
Ð__call__"
_tf_keras_sequentialä{"class_name": "Sequential", "name": "cnn_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 26, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 26, 26, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ò
|layer_with_weights-0
|layer-0
}layer_with_weights-1
}layer-1
~layer-2
	variables
regularization_losses
trainable_variables
	keras_api
+Ñ&call_and_return_all_conditional_losses
Ò__call__"
_tf_keras_sequentialä{"class_name": "Sequential", "name": "cnn_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ø
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
+Ó&call_and_return_all_conditional_losses
Ô__call__"
_tf_keras_sequentialä{"class_name": "Sequential", "name": "cnn_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22, 22, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22, 22, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ø
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
+Õ&call_and_return_all_conditional_losses
Ö__call__"
_tf_keras_sequentialä{"class_name": "Sequential", "name": "cnn_block_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ø
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
+×&call_and_return_all_conditional_losses
Ø__call__"
_tf_keras_sequentialä{"class_name": "Sequential", "name": "cnn_block_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18, 18, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 18, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18, 18, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
ú
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
+Ù&call_and_return_all_conditional_losses
Ú__call__"
_tf_keras_sequentialæ{"class_name": "Sequential", "name": "cnn_block_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "cnn_block_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_6_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "cnn_block_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_6_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}]}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
O	variables
layer_metrics
 layers
¡non_trainable_variables
Pregularization_losses
¢metrics
 £layer_regularization_losses
Qtrainable_variables
Æ__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
X
H0
I1
J2
K3
L4
M5
N6
	7"
trackable_list_wrapper

10
21
32
43
54
65
76
87
98
:9
;10
<11
=12
>13"
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
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
µ
Z	variables
¤layer_metrics
¥layers
¦non_trainable_variables
[regularization_losses
§metrics
 ¨layer_regularization_losses
\trainable_variables
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
`	variables
©layer_metrics
ªlayers
«non_trainable_variables
aregularization_losses
¬metrics
 ­layer_regularization_losses
btrainable_variables
Ê__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
µ
e	variables
®layer_metrics
¯layers
°non_trainable_variables
fregularization_losses
±metrics
 ²layer_regularization_losses
gtrainable_variables
Ì__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper


³_inbound_nodes

kernel
bias
´_outbound_nodes
µ	variables
¶regularization_losses
·trainable_variables
¸	keras_api
+Û&call_and_return_all_conditional_losses
Ü__call__"Ç
_tf_keras_layer­{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 28, 28, 1]}}
å	
¹_inbound_nodes
	ºaxis
	gamma
beta
1moving_mean
2moving_variance
»_outbound_nodes
¼	variables
½regularization_losses
¾trainable_variables
¿	keras_api
+Ý&call_and_return_all_conditional_losses
Þ__call__"ß
_tf_keras_layerÅ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 26, 26, 16]}}

À_inbound_nodes
Á	variables
Âregularization_losses
Ãtrainable_variables
Ä	keras_api
+ß&call_and_return_all_conditional_losses
à__call__"Ø
_tf_keras_layer¾{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
0
1
2
3
14
25"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
µ
q	variables
Ålayer_metrics
Ælayers
Çnon_trainable_variables
rregularization_losses
Èmetrics
 Élayer_regularization_losses
strainable_variables
Î__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
£

Ê_inbound_nodes

kernel
bias
Ë_outbound_nodes
Ì	variables
Íregularization_losses
Îtrainable_variables
Ï	keras_api
+á&call_and_return_all_conditional_losses
â__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 26, 26, 16]}}
é	
Ð_inbound_nodes
	Ñaxis
	gamma
beta
3moving_mean
4moving_variance
Ò_outbound_nodes
Ó	variables
Ôregularization_losses
Õtrainable_variables
Ö	keras_api
+ã&call_and_return_all_conditional_losses
ä__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 24, 24, 32]}}

×_inbound_nodes
Ø	variables
Ùregularization_losses
Útrainable_variables
Û	keras_api
+å&call_and_return_all_conditional_losses
æ__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
0
1
2
3
34
45"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
µ
x	variables
Ülayer_metrics
Ýlayers
Þnon_trainable_variables
yregularization_losses
ßmetrics
 àlayer_regularization_losses
ztrainable_variables
Ð__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
£

á_inbound_nodes

kernel
bias
â_outbound_nodes
ã	variables
äregularization_losses
åtrainable_variables
æ	keras_api
+ç&call_and_return_all_conditional_losses
è__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 24, 24, 32]}}
é	
ç_inbound_nodes
	èaxis
	gamma
 beta
5moving_mean
6moving_variance
é_outbound_nodes
ê	variables
ëregularization_losses
ìtrainable_variables
í	keras_api
+é&call_and_return_all_conditional_losses
ê__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 22, 22, 32]}}

î_inbound_nodes
ï	variables
ðregularization_losses
ñtrainable_variables
ò	keras_api
+ë&call_and_return_all_conditional_losses
ì__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
0
1
2
 3
54
65"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
·
	variables
ólayer_metrics
ôlayers
õnon_trainable_variables
regularization_losses
ömetrics
 ÷layer_regularization_losses
trainable_variables
Ò__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
£

ø_inbound_nodes

!kernel
"bias
ù_outbound_nodes
ú	variables
ûregularization_losses
ütrainable_variables
ý	keras_api
+í&call_and_return_all_conditional_losses
î__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 22, 22, 32]}}
é	
þ_inbound_nodes
	ÿaxis
	#gamma
$beta
7moving_mean
8moving_variance
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+ï&call_and_return_all_conditional_losses
ð__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 20, 20, 32]}}

_inbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+ñ&call_and_return_all_conditional_losses
ò__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
!0
"1
#2
$3
74
85"
trackable_list_wrapper
 "
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
¸
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
Ô__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
£

_inbound_nodes

%kernel
&bias
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 20, 20, 32]}}
é	
_inbound_nodes
	axis
	'gamma
(beta
9moving_mean
:moving_variance
_outbound_nodes
	variables
regularization_losses
trainable_variables
	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 18, 18, 64]}}

_inbound_nodes
	variables
regularization_losses
trainable_variables
 	keras_api
+÷&call_and_return_all_conditional_losses
ø__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
%0
&1
'2
(3
94
:5"
trackable_list_wrapper
 "
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
¸
	variables
¡layer_metrics
¢layers
£non_trainable_variables
regularization_losses
¤metrics
 ¥layer_regularization_losses
trainable_variables
Ö__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
£

¦_inbound_nodes

)kernel
*bias
§_outbound_nodes
¨	variables
©regularization_losses
ªtrainable_variables
«	keras_api
+ù&call_and_return_all_conditional_losses
ú__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 18, 18, 64]}}
é	
¬_inbound_nodes
	­axis
	+gamma
,beta
;moving_mean
<moving_variance
®_outbound_nodes
¯	variables
°regularization_losses
±trainable_variables
²	keras_api
+û&call_and_return_all_conditional_losses
ü__call__"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16, 16, 64]}}

³_inbound_nodes
´	variables
µregularization_losses
¶trainable_variables
·	keras_api
+ý&call_and_return_all_conditional_losses
þ__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
)0
*1
+2
,3
;4
<5"
trackable_list_wrapper
 "
trackable_list_wrapper
<
)0
*1
+2
,3"
trackable_list_wrapper
¸
	variables
¸layer_metrics
¹layers
ºnon_trainable_variables
regularization_losses
»metrics
 ¼layer_regularization_losses
trainable_variables
Ø__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
¤

½_inbound_nodes

-kernel
.bias
¾_outbound_nodes
¿	variables
Àregularization_losses
Átrainable_variables
Â	keras_api
+ÿ&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16, 16, 64]}}
ë	
Ã_inbound_nodes
	Äaxis
	/gamma
0beta
=moving_mean
>moving_variance
Å_outbound_nodes
Æ	variables
Çregularization_losses
Ètrainable_variables
É	keras_api
+&call_and_return_all_conditional_losses
__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 14, 14, 128]}}

Ê_inbound_nodes
Ë	variables
Ìregularization_losses
Ítrainable_variables
Î	keras_api
+&call_and_return_all_conditional_losses
__call__"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
J
-0
.1
/2
03
=4
>5"
trackable_list_wrapper
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
¸
	variables
Ïlayer_metrics
Ðlayers
Ñnon_trainable_variables
regularization_losses
Òmetrics
 Ólayer_regularization_losses
trainable_variables
Ú__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
µ	variables
Ôlayer_metrics
Õlayers
Önon_trainable_variables
¶regularization_losses
×metrics
 Ølayer_regularization_losses
·trainable_variables
Ü__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
¼	variables
Ùlayer_metrics
Úlayers
Ûnon_trainable_variables
½regularization_losses
Ümetrics
 Ýlayer_regularization_losses
¾trainable_variables
Þ__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
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
Á	variables
Þlayer_metrics
ßlayers
ànon_trainable_variables
Âregularization_losses
ámetrics
 âlayer_regularization_losses
Ãtrainable_variables
à__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
n0
o1
p2"
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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
Ì	variables
ãlayer_metrics
älayers
ånon_trainable_variables
Íregularization_losses
æmetrics
 çlayer_regularization_losses
Îtrainable_variables
â__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
32
43"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
Ó	variables
èlayer_metrics
élayers
ênon_trainable_variables
Ôregularization_losses
ëmetrics
 ìlayer_regularization_losses
Õtrainable_variables
ä__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
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
Ø	variables
ílayer_metrics
îlayers
ïnon_trainable_variables
Ùregularization_losses
ðmetrics
 ñlayer_regularization_losses
Útrainable_variables
æ__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
u0
v1
w2"
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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
¸
ã	variables
òlayer_metrics
ólayers
ônon_trainable_variables
äregularization_losses
õmetrics
 ölayer_regularization_losses
åtrainable_variables
è__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
 1
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
¸
ê	variables
÷layer_metrics
ølayers
ùnon_trainable_variables
ëregularization_losses
úmetrics
 ûlayer_regularization_losses
ìtrainable_variables
ê__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
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
ï	variables
ülayer_metrics
ýlayers
þnon_trainable_variables
ðregularization_losses
ÿmetrics
 layer_regularization_losses
ñtrainable_variables
ì__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
5
|0
}1
~2"
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
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
¸
ú	variables
layer_metrics
layers
non_trainable_variables
ûregularization_losses
metrics
 layer_regularization_losses
ütrainable_variables
î__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
#0
$1
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
¸
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
ð__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
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
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
ò__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
8
0
1
2"
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
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
¸
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
'0
(1
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
¸
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
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
	variables
layer_metrics
layers
non_trainable_variables
regularization_losses
metrics
 layer_regularization_losses
trainable_variables
ø__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
8
0
1
2"
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
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
¸
¨	variables
layer_metrics
 layers
¡non_trainable_variables
©regularization_losses
¢metrics
 £layer_regularization_losses
ªtrainable_variables
ú__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
+0
,1
;2
<3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
¸
¯	variables
¤layer_metrics
¥layers
¦non_trainable_variables
°regularization_losses
§metrics
 ¨layer_regularization_losses
±trainable_variables
ü__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
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
´	variables
©layer_metrics
ªlayers
«non_trainable_variables
µregularization_losses
¬metrics
 ­layer_regularization_losses
¶trainable_variables
þ__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
8
0
1
2"
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
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
¸
¿	variables
®layer_metrics
¯layers
°non_trainable_variables
Àregularization_losses
±metrics
 ²layer_regularization_losses
Átrainable_variables
__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
/0
01
=2
>3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
¸
Æ	variables
³layer_metrics
´layers
µnon_trainable_variables
Çregularization_losses
¶metrics
 ·layer_regularization_losses
Ètrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
Ë	variables
¸layer_metrics
¹layers
ºnon_trainable_variables
Ìregularization_losses
»metrics
 ¼layer_regularization_losses
Ítrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
8
0
1
2"
trackable_list_wrapper
.
=0
>1"
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
=0
>1"
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
Ú2×
F__inference_class_cnn_layer_call_and_return_conditional_losses_6964050
F__inference_class_cnn_layer_call_and_return_conditional_losses_6964219
F__inference_class_cnn_layer_call_and_return_conditional_losses_6963497
F__inference_class_cnn_layer_call_and_return_conditional_losses_6963666´
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
î2ë
+__inference_class_cnn_layer_call_fn_6963763
+__inference_class_cnn_layer_call_fn_6964413
+__inference_class_cnn_layer_call_fn_6963860
+__inference_class_cnn_layer_call_fn_6964316´
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
"__inference__wrapped_model_6959242¾
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
2
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6965231
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6964580
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6965078
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6964733°
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
2
7__inference_feature_extractor_cnn_layer_call_fn_6965320
7__inference_feature_extractor_cnn_layer_call_fn_6964911
7__inference_feature_extractor_cnn_layer_call_fn_6964822
7__inference_feature_extractor_cnn_layer_call_fn_6965409°
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
ò2ï
I__inference_feed_forward_layer_call_and_return_conditional_losses_6965503
I__inference_feed_forward_layer_call_and_return_conditional_losses_6965521
I__inference_feed_forward_layer_call_and_return_conditional_losses_6965434
I__inference_feed_forward_layer_call_and_return_conditional_losses_6965452À
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
2
.__inference_feed_forward_layer_call_fn_6965534
.__inference_feed_forward_layer_call_fn_6965547
.__inference_feed_forward_layer_call_fn_6965465
.__inference_feed_forward_layer_call_fn_6965478À
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
4B2
%__inference_signature_wrapper_6963307input_1
½2º
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_6961440à
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
¢2
:__inference_global_average_pooling2d_layer_call_fn_6961446à
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
ì2é
B__inference_dense_layer_call_and_return_conditional_losses_6965558¢
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
Ñ2Î
'__inference_dense_layer_call_fn_6965567¢
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
Æ2Ã
D__inference_dropout_layer_call_and_return_conditional_losses_6965579
D__inference_dropout_layer_call_and_return_conditional_losses_6965584´
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
2
)__inference_dropout_layer_call_fn_6965594
)__inference_dropout_layer_call_fn_6965589´
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
D__inference_dense_1_layer_call_and_return_conditional_losses_6965604¢
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
)__inference_dense_1_layer_call_fn_6965613¢
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
î2ë
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965751
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965665
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965726
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965640À
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
-__inference_cnn_block_0_layer_call_fn_6965682
-__inference_cnn_block_0_layer_call_fn_6965785
-__inference_cnn_block_0_layer_call_fn_6965768
-__inference_cnn_block_0_layer_call_fn_6965699À
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
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965837
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965923
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965812
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965898À
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
-__inference_cnn_block_1_layer_call_fn_6965940
-__inference_cnn_block_1_layer_call_fn_6965854
-__inference_cnn_block_1_layer_call_fn_6965871
-__inference_cnn_block_1_layer_call_fn_6965957À
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
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966009
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966070
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966095
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6965984À
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
-__inference_cnn_block_2_layer_call_fn_6966043
-__inference_cnn_block_2_layer_call_fn_6966112
-__inference_cnn_block_2_layer_call_fn_6966026
-__inference_cnn_block_2_layer_call_fn_6966129À
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
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966267
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966181
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966242
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966156À
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
-__inference_cnn_block_3_layer_call_fn_6966301
-__inference_cnn_block_3_layer_call_fn_6966215
-__inference_cnn_block_3_layer_call_fn_6966198
-__inference_cnn_block_3_layer_call_fn_6966284À
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
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966328
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966353
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966414
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966439À
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
-__inference_cnn_block_4_layer_call_fn_6966456
-__inference_cnn_block_4_layer_call_fn_6966473
-__inference_cnn_block_4_layer_call_fn_6966370
-__inference_cnn_block_4_layer_call_fn_6966387À
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
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966500
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966611
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966525
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966586À
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
-__inference_cnn_block_5_layer_call_fn_6966628
-__inference_cnn_block_5_layer_call_fn_6966559
-__inference_cnn_block_5_layer_call_fn_6966645
-__inference_cnn_block_5_layer_call_fn_6966542À
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
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966697
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966672
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966758
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966783À
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
-__inference_cnn_block_6_layer_call_fn_6966800
-__inference_cnn_block_6_layer_call_fn_6966817
-__inference_cnn_block_6_layer_call_fn_6966714
-__inference_cnn_block_6_layer_call_fn_6966731À
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
í2ê
C__inference_conv2d_layer_call_and_return_conditional_losses_6966827¢
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
Ò2Ï
(__inference_conv2d_layer_call_fn_6966836¢
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
2ÿ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966856
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966920
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966874
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966938´
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
2
5__inference_batch_normalization_layer_call_fn_6966900
5__inference_batch_normalization_layer_call_fn_6966887
5__inference_batch_normalization_layer_call_fn_6966964
5__inference_batch_normalization_layer_call_fn_6966951´
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
ì2é
B__inference_re_lu_layer_call_and_return_conditional_losses_6966969¢
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
Ñ2Î
'__inference_re_lu_layer_call_fn_6966974¢
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
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6966984¢
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
*__inference_conv2d_1_layer_call_fn_6966993¢
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967013
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967031
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967095
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967077´
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
7__inference_batch_normalization_1_layer_call_fn_6967121
7__inference_batch_normalization_1_layer_call_fn_6967044
7__inference_batch_normalization_1_layer_call_fn_6967057
7__inference_batch_normalization_1_layer_call_fn_6967108´
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
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6967126¢
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
)__inference_re_lu_1_layer_call_fn_6967131¢
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
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6967141¢
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
*__inference_conv2d_2_layer_call_fn_6967150¢
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967170
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967188
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967234
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967252´
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
7__inference_batch_normalization_2_layer_call_fn_6967278
7__inference_batch_normalization_2_layer_call_fn_6967201
7__inference_batch_normalization_2_layer_call_fn_6967214
7__inference_batch_normalization_2_layer_call_fn_6967265´
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
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6967283¢
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
)__inference_re_lu_2_layer_call_fn_6967288¢
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
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6967298¢
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
*__inference_conv2d_3_layer_call_fn_6967307¢
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967345
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967391
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967327
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967409´
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
7__inference_batch_normalization_3_layer_call_fn_6967358
7__inference_batch_normalization_3_layer_call_fn_6967435
7__inference_batch_normalization_3_layer_call_fn_6967371
7__inference_batch_normalization_3_layer_call_fn_6967422´
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
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6967440¢
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
)__inference_re_lu_3_layer_call_fn_6967445¢
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
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6967455¢
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
*__inference_conv2d_4_layer_call_fn_6967464¢
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967484
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967566
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967548
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967502´
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
7__inference_batch_normalization_4_layer_call_fn_6967579
7__inference_batch_normalization_4_layer_call_fn_6967592
7__inference_batch_normalization_4_layer_call_fn_6967528
7__inference_batch_normalization_4_layer_call_fn_6967515´
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
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6967597¢
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
)__inference_re_lu_4_layer_call_fn_6967602¢
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
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6967612¢
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
*__inference_conv2d_5_layer_call_fn_6967621¢
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967723
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967641
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967705
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967659´
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
7__inference_batch_normalization_5_layer_call_fn_6967672
7__inference_batch_normalization_5_layer_call_fn_6967685
7__inference_batch_normalization_5_layer_call_fn_6967749
7__inference_batch_normalization_5_layer_call_fn_6967736´
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
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6967754¢
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
)__inference_re_lu_5_layer_call_fn_6967759¢
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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6967769¢
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
*__inference_conv2d_6_layer_call_fn_6967778¢
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967798
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967880
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967862
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967816´
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
7__inference_batch_normalization_6_layer_call_fn_6967829
7__inference_batch_normalization_6_layer_call_fn_6967906
7__inference_batch_normalization_6_layer_call_fn_6967842
7__inference_batch_normalization_6_layer_call_fn_6967893´
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
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6967911¢
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
)__inference_re_lu_6_layer_call_fn_6967916¢
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
 Æ
"__inference__wrapped_model_6959242.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
í
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_696701334M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 í
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_696703134M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967077r34;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6967095r34;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Å
7__inference_batch_normalization_1_layer_call_fn_696704434M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Å
7__inference_batch_normalization_1_layer_call_fn_696705734M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_1_layer_call_fn_6967108e34;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_1_layer_call_fn_6967121e34;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ í
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967170 56M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 í
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967188 56M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967234r 56;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6967252r 56;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Å
7__inference_batch_normalization_2_layer_call_fn_6967201 56M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Å
7__inference_batch_normalization_2_layer_call_fn_6967214 56M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_2_layer_call_fn_6967265e 56;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_2_layer_call_fn_6967278e 56;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ í
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967327#$78M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 í
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967345#$78M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967391r#$78;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 È
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6967409r#$78;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Å
7__inference_batch_normalization_3_layer_call_fn_6967358#$78M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Å
7__inference_batch_normalization_3_layer_call_fn_6967371#$78M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_3_layer_call_fn_6967422e#$78;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ  
7__inference_batch_normalization_3_layer_call_fn_6967435e#$78;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ È
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967484r'(9:;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 È
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967502r'(9:;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 í
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967548'(9:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 í
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6967566'(9:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
  
7__inference_batch_normalization_4_layer_call_fn_6967515e'(9:;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@ 
7__inference_batch_normalization_4_layer_call_fn_6967528e'(9:;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@Å
7__inference_batch_normalization_4_layer_call_fn_6967579'(9:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Å
7__inference_batch_normalization_4_layer_call_fn_6967592'(9:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@í
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967641+,;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 í
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967659+,;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 È
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967705r+,;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 È
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6967723r+,;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Å
7__inference_batch_normalization_5_layer_call_fn_6967672+,;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Å
7__inference_batch_normalization_5_layer_call_fn_6967685+,;<M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ 
7__inference_batch_normalization_5_layer_call_fn_6967736e+,;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@ 
7__inference_batch_normalization_5_layer_call_fn_6967749e+,;<;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@ï
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967798/0=>N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967816/0=>N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967862t/0=><¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6967880t/0=><¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ç
7__inference_batch_normalization_6_layer_call_fn_6967829/0=>N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
7__inference_batch_normalization_6_layer_call_fn_6967842/0=>N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_6_layer_call_fn_6967893g/0=><¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_6_layer_call_fn_6967906g/0=><¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿë
P__inference_batch_normalization_layer_call_and_return_conditional_losses_696685612M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ë
P__inference_batch_normalization_layer_call_and_return_conditional_losses_696687412M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966920r12;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Æ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_6966938r12;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ã
5__inference_batch_normalization_layer_call_fn_696688712M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
5__inference_batch_normalization_layer_call_fn_696690012M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5__inference_batch_normalization_layer_call_fn_6966951e12;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ
5__inference_batch_normalization_layer_call_fn_6966964e12;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿà
F__inference_class_cnn_layer_call_and_return_conditional_losses_6963497.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 à
F__inference_class_cnn_layer_call_and_return_conditional_losses_6963666.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ß
F__inference_class_cnn_layer_call_and_return_conditional_losses_6964050.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ß
F__inference_class_cnn_layer_call_and_return_conditional_losses_6964219.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¸
+__inference_class_cnn_layer_call_fn_6963763.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
¸
+__inference_class_cnn_layer_call_fn_6963860.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
·
+__inference_class_cnn_layer_call_fn_6964316.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
·
+__inference_class_cnn_layer_call_fn_6964413.1234 56!"#$78%&'(9:)*+,;<-./0=>?@AB;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
Ä
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965640x12?¢<
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
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965665x12?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ê
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965726~12E¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ê
H__inference_cnn_block_0_layer_call_and_return_conditional_losses_6965751~12E¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_cnn_block_0_layer_call_fn_6965682k12?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ
-__inference_cnn_block_0_layer_call_fn_6965699k12?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ¢
-__inference_cnn_block_0_layer_call_fn_6965768q12E¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ¢
-__inference_cnn_block_0_layer_call_fn_6965785q12E¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿÍ
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_696581234G¢D
=¢:
0-
conv2d_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Í
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_696583734G¢D
=¢:
0-
conv2d_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ä
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965898x34?¢<
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
H__inference_cnn_block_1_layer_call_and_return_conditional_losses_6965923x34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ¤
-__inference_cnn_block_1_layer_call_fn_6965854s34G¢D
=¢:
0-
conv2d_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ ¤
-__inference_cnn_block_1_layer_call_fn_6965871s34G¢D
=¢:
0-
conv2d_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_1_layer_call_fn_6965940k34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_1_layer_call_fn_6965957k34?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ Í
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6965984 56G¢D
=¢:
0-
conv2d_2_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Í
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966009 56G¢D
=¢:
0-
conv2d_2_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ä
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966070x 56?¢<
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
H__inference_cnn_block_2_layer_call_and_return_conditional_losses_6966095x 56?¢<
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
-__inference_cnn_block_2_layer_call_fn_6966026s 56G¢D
=¢:
0-
conv2d_2_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ ¤
-__inference_cnn_block_2_layer_call_fn_6966043s 56G¢D
=¢:
0-
conv2d_2_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_2_layer_call_fn_6966112k 56?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_2_layer_call_fn_6966129k 56?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ Í
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966156!"#$78G¢D
=¢:
0-
conv2d_3_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Í
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966181!"#$78G¢D
=¢:
0-
conv2d_3_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ä
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966242x!"#$78?¢<
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
H__inference_cnn_block_3_layer_call_and_return_conditional_losses_6966267x!"#$78?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ¤
-__inference_cnn_block_3_layer_call_fn_6966198s!"#$78G¢D
=¢:
0-
conv2d_3_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ ¤
-__inference_cnn_block_3_layer_call_fn_6966215s!"#$78G¢D
=¢:
0-
conv2d_3_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_3_layer_call_fn_6966284k!"#$78?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ 
-__inference_cnn_block_3_layer_call_fn_6966301k!"#$78?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ Ä
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966328x%&'(9:?¢<
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
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966353x%&'(9:?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Í
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966414%&'(9:G¢D
=¢:
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Í
H__inference_cnn_block_4_layer_call_and_return_conditional_losses_6966439%&'(9:G¢D
=¢:
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_cnn_block_4_layer_call_fn_6966370k%&'(9:?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ@
-__inference_cnn_block_4_layer_call_fn_6966387k%&'(9:?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@¤
-__inference_cnn_block_4_layer_call_fn_6966456s%&'(9:G¢D
=¢:
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª " ÿÿÿÿÿÿÿÿÿ@¤
-__inference_cnn_block_4_layer_call_fn_6966473s%&'(9:G¢D
=¢:
0-
conv2d_4_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@Í
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966500)*+,;<G¢D
=¢:
0-
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Í
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966525)*+,;<G¢D
=¢:
0-
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ä
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966586x)*+,;<?¢<
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
H__inference_cnn_block_5_layer_call_and_return_conditional_losses_6966611x)*+,;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ¤
-__inference_cnn_block_5_layer_call_fn_6966542s)*+,;<G¢D
=¢:
0-
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª " ÿÿÿÿÿÿÿÿÿ@¤
-__inference_cnn_block_5_layer_call_fn_6966559s)*+,;<G¢D
=¢:
0-
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@
-__inference_cnn_block_5_layer_call_fn_6966628k)*+,;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª " ÿÿÿÿÿÿÿÿÿ@
-__inference_cnn_block_5_layer_call_fn_6966645k)*+,;<?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@Î
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966672-./0=>G¢D
=¢:
0-
conv2d_6_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Î
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966697-./0=>G¢D
=¢:
0-
conv2d_6_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Å
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966758y-./0=>?¢<
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
H__inference_cnn_block_6_layer_call_and_return_conditional_losses_6966783y-./0=>?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¥
-__inference_cnn_block_6_layer_call_fn_6966714t-./0=>G¢D
=¢:
0-
conv2d_6_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ¥
-__inference_cnn_block_6_layer_call_fn_6966731t-./0=>G¢D
=¢:
0-
conv2d_6_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ
-__inference_cnn_block_6_layer_call_fn_6966800l-./0=>?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ
-__inference_cnn_block_6_layer_call_fn_6966817l-./0=>?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿµ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_6966984l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv2d_1_layer_call_fn_6966993_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ µ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_6967141l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv2d_2_layer_call_fn_6967150_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ µ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_6967298l!"7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv2d_3_layer_call_fn_6967307_!"7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ µ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_6967455l%&7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_4_layer_call_fn_6967464_%&7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@µ
E__inference_conv2d_5_layer_call_and_return_conditional_losses_6967612l)*7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_5_layer_call_fn_6967621_)*7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¶
E__inference_conv2d_6_layer_call_and_return_conditional_losses_6967769m-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_6_layer_call_fn_6967778`-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ³
C__inference_conv2d_layer_call_and_return_conditional_losses_6966827l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
(__inference_conv2d_layer_call_fn_6966836_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_1_layer_call_and_return_conditional_losses_6965604\AB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 |
)__inference_dense_1_layer_call_fn_6965613OAB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
£
B__inference_dense_layer_call_and_return_conditional_losses_6965558]?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
'__inference_dense_layer_call_fn_6965567P?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dropout_layer_call_and_return_conditional_losses_6965579\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¤
D__inference_dropout_layer_call_and_return_conditional_losses_6965584\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dropout_layer_call_fn_6965589O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@|
)__inference_dropout_layer_call_fn_6965594O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@è
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6964580*1234 56!"#$78%&'(9:)*+,;<-./0=>;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 è
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6964733*1234 56!"#$78%&'(9:)*+,;<-./0=>;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 é
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6965078*1234 56!"#$78%&'(9:)*+,;<-./0=><¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 é
R__inference_feature_extractor_cnn_layer_call_and_return_conditional_losses_6965231*1234 56!"#$78%&'(9:)*+,;<-./0=><¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 À
7__inference_feature_extractor_cnn_layer_call_fn_6964822*1234 56!"#$78%&'(9:)*+,;<-./0=>;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÀ
7__inference_feature_extractor_cnn_layer_call_fn_6964911*1234 56!"#$78%&'(9:)*+,;<-./0=>;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿÁ
7__inference_feature_extractor_cnn_layer_call_fn_6965320*1234 56!"#$78%&'(9:)*+,;<-./0=><¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÁ
7__inference_feature_extractor_cnn_layer_call_fn_6965409*1234 56!"#$78%&'(9:)*+,;<-./0=><¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¹
I__inference_feed_forward_layer_call_and_return_conditional_losses_6965434l?@AB=¢:
3¢0
&#
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¹
I__inference_feed_forward_layer_call_and_return_conditional_losses_6965452l?@AB=¢:
3¢0
&#
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ´
I__inference_feed_forward_layer_call_and_return_conditional_losses_6965503g?@AB8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ´
I__inference_feed_forward_layer_call_and_return_conditional_losses_6965521g?@AB8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
.__inference_feed_forward_layer_call_fn_6965465_?@AB=¢:
3¢0
&#
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_feed_forward_layer_call_fn_6965478_?@AB=¢:
3¢0
&#
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_feed_forward_layer_call_fn_6965534Z?@AB8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_feed_forward_layer_call_fn_6965547Z?@AB8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Þ
U__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_6961440R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
:__inference_global_average_pooling2d_layer_call_fn_6961446wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6967126h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_re_lu_1_layer_call_fn_6967131[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ °
D__inference_re_lu_2_layer_call_and_return_conditional_losses_6967283h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_re_lu_2_layer_call_fn_6967288[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ °
D__inference_re_lu_3_layer_call_and_return_conditional_losses_6967440h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_re_lu_3_layer_call_fn_6967445[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ °
D__inference_re_lu_4_layer_call_and_return_conditional_losses_6967597h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_re_lu_4_layer_call_fn_6967602[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@°
D__inference_re_lu_5_layer_call_and_return_conditional_losses_6967754h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_re_lu_5_layer_call_fn_6967759[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@²
D__inference_re_lu_6_layer_call_and_return_conditional_losses_6967911j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_re_lu_6_layer_call_fn_6967916]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ®
B__inference_re_lu_layer_call_and_return_conditional_losses_6966969h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
'__inference_re_lu_layer_call_fn_6966974[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÔ
%__inference_signature_wrapper_6963307ª.1234 56!"#$78%&'(9:)*+,;<-./0=>?@ABC¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
