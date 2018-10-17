	.text
	.file	"default_function"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4
.LCPI0_0:
	.quad	16
	.quad	1
	.text
	.globl	default_function
	.p2align	4, 0x90
	.type	default_function,@function
default_function:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	cmpl	$3, %edx
	jne	.LBB0_1
	movq	(%rdi), %rax
	movq	8(%rdi), %rcx
	movl	(%rsi), %ebp
	movl	4(%rsi), %r15d
	movq	16(%rdi), %r10
	movl	8(%rsi), %r14d
	movq	(%rax), %rsi
	movq	24(%rax), %r13
	movq	32(%rax), %rdx
	testq	%rdx, %rdx
	je	.LBB0_9
	cmpl	$1605632, (%rdx)
	jne	.LBB0_98
	cmpl	$200704, 8(%rdx)
	jne	.LBB0_98
	cmpl	$1792, 16(%rdx)
	jne	.LBB0_98
	vpxor	%xmm0, %xmm0, %xmm0
	vpblendd	$5, 24(%rdx), %xmm0, %xmm0
	vpcmpeqq	.LCPI0_0(%rip), %xmm0, %k0
	kmovd	%k0, %edx
	testb	$1, %dl
	je	.LBB0_98
	kshiftrw	$1, %k0, %k0
	kmovd	%k0, %edx
	testb	$1, %dl
	je	.LBB0_98
.LBB0_9:
	movl	8(%rax), %r12d
	movl	12(%rax), %r8d
	movq	(%rcx), %rdx
	movq	24(%rcx), %r11
	movq	32(%rcx), %rdi
	testq	%rdi, %rdi
	je	.LBB0_15
	cmpl	$831744, (%rdi)
	jne	.LBB0_99
	cmpl	$207936, 8(%rdi)
	jne	.LBB0_99
	cmpl	$1824, 16(%rdi)
	jne	.LBB0_99
	vpxor	%xmm0, %xmm0, %xmm0
	vpblendd	$5, 24(%rdi), %xmm0, %xmm0
	vpcmpeqq	.LCPI0_0(%rip), %xmm0, %k0
	kmovd	%k0, %edi
	testb	$1, %dil
	je	.LBB0_99
	kshiftrw	$1, %k0, %k0
	kmovd	%k0, %edi
	testb	$1, %dil
	je	.LBB0_99
.LBB0_15:
	movq	(%r10), %rdi
	movq	24(%r10), %r9
	movq	32(%r10), %rbx
	testq	%rbx, %rbx
	je	.LBB0_22
	cmpl	$9216, (%rbx)
	jne	.LBB0_100
	cmpl	$2304, 8(%rbx)
	jne	.LBB0_100
	cmpl	$768, 16(%rbx)
	jne	.LBB0_100
	cmpl	$256, 24(%rbx)
	jne	.LBB0_100
	vpxor	%xmm0, %xmm0, %xmm0
	vpblendd	$5, 32(%rbx), %xmm0, %xmm0
	vpcmpeqq	.LCPI0_0(%rip), %xmm0, %k0
	kmovd	%k0, %ebx
	testb	$1, %bl
	je	.LBB0_100
	kshiftrw	$1, %k0, %k0
	kmovd	%k0, %ebx
	testb	$1, %bl
	je	.LBB0_100
.LBB0_22:
	cmpl	$7, %ebp
	ja	.LBB0_24
	movl	$152, %ebx
	btl	%ebp, %ebx
	jae	.LBB0_24
	cmpl	$7, %r15d
	ja	.LBB0_27
	movl	$152, %ebx
	btl	%r15d, %ebx
	jae	.LBB0_27
	cmpl	$7, %r14d
	ja	.LBB0_30
	movl	$152, %ebx
	btl	%r14d, %ebx
	jae	.LBB0_30
	cmpl	$1, %r12d
	jne	.LBB0_32
	cmpl	$5, 16(%rax)
	jne	.LBB0_34
	movzwl	22(%rax), %ebx
	cmpl	$1, %ebx
	jne	.LBB0_38
	cmpb	$32, 21(%rax)
	jne	.LBB0_38
	cmpb	$2, 20(%rax)
	jne	.LBB0_38
	cmpl	$64, (%r13)
	jne	.LBB0_40
	cmpl	$8, 8(%r13)
	jne	.LBB0_42
	cmpl	$112, 16(%r13)
	jne	.LBB0_44
	cmpl	$112, 24(%r13)
	jne	.LBB0_46
	cmpl	$16, 32(%r13)
	jne	.LBB0_48
	cmpq	$0, 40(%rax)
	jne	.LBB0_50
	cmpl	$5, 16(%rcx)
	jne	.LBB0_52
	movzwl	22(%rcx), %eax
	cmpl	$1, %eax
	jne	.LBB0_56
	cmpb	$32, 21(%rcx)
	jne	.LBB0_56
	cmpb	$2, 20(%rcx)
	jne	.LBB0_56
	cmpl	$64, (%r11)
	jne	.LBB0_58
	cmpl	$4, 8(%r11)
	jne	.LBB0_60
	cmpl	$114, 16(%r11)
	jne	.LBB0_62
	cmpl	$114, 24(%r11)
	jne	.LBB0_64
	cmpl	$16, 32(%r11)
	jne	.LBB0_66
	cmpq	$0, 40(%rcx)
	jne	.LBB0_68
	cmpl	$1, 8(%rcx)
	jne	.LBB0_70
	cmpl	12(%rcx), %r8d
	jne	.LBB0_72
	cmpl	$6, 16(%r10)
	jne	.LBB0_74
	movzwl	22(%r10), %eax
	cmpl	$1, %eax
	jne	.LBB0_78
	cmpb	$32, 21(%r10)
	jne	.LBB0_78
	cmpb	$2, 20(%r10)
	jne	.LBB0_78
	cmpl	$8, (%r9)
	jne	.LBB0_80
	cmpl	$4, 8(%r9)
	jne	.LBB0_82
	cmpl	$3, 16(%r9)
	jne	.LBB0_84
	cmpl	$3, 24(%r9)
	jne	.LBB0_86
	cmpl	$16, 32(%r9)
	jne	.LBB0_88
	cmpl	$16, 40(%r9)
	jne	.LBB0_90
	cmpq	$0, 40(%r10)
	jne	.LBB0_92
	cmpl	$1, 8(%r10)
	jne	.LBB0_94
	cmpl	12(%r10), %r8d
	jne	.LBB0_96
	addq	$8, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	jmp	.Ldefault_function_compute_
.LBB0_24:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.4(%rip), %rdi
	jmp	.LBB0_2
.LBB0_27:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.5(%rip), %rdi
	jmp	.LBB0_2
.LBB0_30:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.6(%rip), %rdi
.LBB0_2:
	callq	*(%rax)
	movl	$-1, %eax
	addq	$8, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.LBB0_1:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str(%rip), %rdi
	jmp	.LBB0_2
.LBB0_98:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.1(%rip), %rdi
	jmp	.LBB0_2
.LBB0_99:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.2(%rip), %rdi
	jmp	.LBB0_2
.LBB0_100:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.3(%rip), %rdi
	jmp	.LBB0_2
.LBB0_32:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.7(%rip), %rdi
	jmp	.LBB0_2
.LBB0_34:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.8(%rip), %rdi
	jmp	.LBB0_2
.LBB0_38:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.9(%rip), %rdi
	jmp	.LBB0_2
.LBB0_40:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.10(%rip), %rdi
	jmp	.LBB0_2
.LBB0_42:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.11(%rip), %rdi
	jmp	.LBB0_2
.LBB0_44:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.12(%rip), %rdi
	jmp	.LBB0_2
.LBB0_46:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.13(%rip), %rdi
	jmp	.LBB0_2
.LBB0_48:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.14(%rip), %rdi
	jmp	.LBB0_2
.LBB0_50:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.15(%rip), %rdi
	jmp	.LBB0_2
.LBB0_52:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.16(%rip), %rdi
	jmp	.LBB0_2
.LBB0_56:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.17(%rip), %rdi
	jmp	.LBB0_2
.LBB0_58:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.18(%rip), %rdi
	jmp	.LBB0_2
.LBB0_60:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.19(%rip), %rdi
	jmp	.LBB0_2
.LBB0_62:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.20(%rip), %rdi
	jmp	.LBB0_2
.LBB0_64:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.21(%rip), %rdi
	jmp	.LBB0_2
.LBB0_66:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.22(%rip), %rdi
	jmp	.LBB0_2
.LBB0_68:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.23(%rip), %rdi
	jmp	.LBB0_2
.LBB0_70:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.24(%rip), %rdi
	jmp	.LBB0_2
.LBB0_72:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.25(%rip), %rdi
	jmp	.LBB0_2
.LBB0_74:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.26(%rip), %rdi
	jmp	.LBB0_2
.LBB0_78:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.27(%rip), %rdi
	jmp	.LBB0_2
.LBB0_80:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.28(%rip), %rdi
	jmp	.LBB0_2
.LBB0_82:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.29(%rip), %rdi
	jmp	.LBB0_2
.LBB0_84:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.30(%rip), %rdi
	jmp	.LBB0_2
.LBB0_86:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.31(%rip), %rdi
	jmp	.LBB0_2
.LBB0_88:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.32(%rip), %rdi
	jmp	.LBB0_2
.LBB0_90:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.33(%rip), %rdi
	jmp	.LBB0_2
.LBB0_92:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.34(%rip), %rdi
	jmp	.LBB0_2
.LBB0_94:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.35(%rip), %rdi
	jmp	.LBB0_2
.LBB0_96:
	movq	__TVMAPISetLastError@GOTPCREL(%rip), %rax
	leaq	.L.str.36(%rip), %rdi
	jmp	.LBB0_2
.Lfunc_end0:
	.size	default_function, .Lfunc_end0-default_function
	.cfi_endproc

	.p2align	4, 0x90
	.type	.Ldefault_function_compute_,@function
.Ldefault_function_compute_:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movq	%rdi, (%rsp)
	movq	%rsi, 8(%rsp)
	movq	%rdx, 16(%rsp)
	movq	__TVMBackendParallelLaunch@GOTPCREL(%rip), %rax
	leaq	.L__tvm_parallel_lambda(%rip), %rdi
	movq	%rsp, %rsi
	xorl	%edx, %edx
	callq	*(%rax)
	addq	$24, %rsp
	retq
.Lfunc_end1:
	.size	.Ldefault_function_compute_, .Lfunc_end1-.Ldefault_function_compute_
	.cfi_endproc

	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5
.LCPI2_0:
	.long	16
	.long	32
	.long	48
	.long	64
	.long	80
	.long	96
	.long	112
	.long	128
.LCPI2_2:
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.long	8
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4
.LCPI2_1:
	.long	144
	.long	160
	.long	176
	.long	192
.LCPI2_3:
	.long	9
	.long	10
	.long	11
	.long	12
	.section	.rodata,"a",@progbits
	.p2align	6
.LCPI2_4:
	.long	1792
	.long	1793
	.long	1794
	.long	1795
	.long	1796
	.long	1797
	.long	1798
	.long	1799
	.long	1800
	.long	1801
	.long	1802
	.long	1803
	.long	1804
	.long	1805
	.long	1806
	.long	1807
.LCPI2_5:
	.long	3584
	.long	3585
	.long	3586
	.long	3587
	.long	3588
	.long	3589
	.long	3590
	.long	3591
	.long	3592
	.long	3593
	.long	3594
	.long	3595
	.long	3596
	.long	3597
	.long	3598
	.long	3599
.LCPI2_6:
	.long	5376
	.long	5377
	.long	5378
	.long	5379
	.long	5380
	.long	5381
	.long	5382
	.long	5383
	.long	5384
	.long	5385
	.long	5386
	.long	5387
	.long	5388
	.long	5389
	.long	5390
	.long	5391
.LCPI2_7:
	.long	7168
	.long	7169
	.long	7170
	.long	7171
	.long	7172
	.long	7173
	.long	7174
	.long	7175
	.long	7176
	.long	7177
	.long	7178
	.long	7179
	.long	7180
	.long	7181
	.long	7182
	.long	7183
.LCPI2_8:
	.long	8960
	.long	8961
	.long	8962
	.long	8963
	.long	8964
	.long	8965
	.long	8966
	.long	8967
	.long	8968
	.long	8969
	.long	8970
	.long	8971
	.long	8972
	.long	8973
	.long	8974
	.long	8975
.LCPI2_9:
	.long	10752
	.long	10753
	.long	10754
	.long	10755
	.long	10756
	.long	10757
	.long	10758
	.long	10759
	.long	10760
	.long	10761
	.long	10762
	.long	10763
	.long	10764
	.long	10765
	.long	10766
	.long	10767
.LCPI2_10:
	.long	12544
	.long	12545
	.long	12546
	.long	12547
	.long	12548
	.long	12549
	.long	12550
	.long	12551
	.long	12552
	.long	12553
	.long	12554
	.long	12555
	.long	12556
	.long	12557
	.long	12558
	.long	12559
.LCPI2_11:
	.long	14336
	.long	14337
	.long	14338
	.long	14339
	.long	14340
	.long	14341
	.long	14342
	.long	14343
	.long	14344
	.long	14345
	.long	14346
	.long	14347
	.long	14348
	.long	14349
	.long	14350
	.long	14351
.LCPI2_12:
	.long	16128
	.long	16129
	.long	16130
	.long	16131
	.long	16132
	.long	16133
	.long	16134
	.long	16135
	.long	16136
	.long	16137
	.long	16138
	.long	16139
	.long	16140
	.long	16141
	.long	16142
	.long	16143
.LCPI2_13:
	.long	17920
	.long	17921
	.long	17922
	.long	17923
	.long	17924
	.long	17925
	.long	17926
	.long	17927
	.long	17928
	.long	17929
	.long	17930
	.long	17931
	.long	17932
	.long	17933
	.long	17934
	.long	17935
.LCPI2_14:
	.long	19712
	.long	19713
	.long	19714
	.long	19715
	.long	19716
	.long	19717
	.long	19718
	.long	19719
	.long	19720
	.long	19721
	.long	19722
	.long	19723
	.long	19724
	.long	19725
	.long	19726
	.long	19727
.LCPI2_15:
	.long	21504
	.long	21505
	.long	21506
	.long	21507
	.long	21508
	.long	21509
	.long	21510
	.long	21511
	.long	21512
	.long	21513
	.long	21514
	.long	21515
	.long	21516
	.long	21517
	.long	21518
	.long	21519
.LCPI2_16:
	.long	23296
	.long	23297
	.long	23298
	.long	23299
	.long	23300
	.long	23301
	.long	23302
	.long	23303
	.long	23304
	.long	23305
	.long	23306
	.long	23307
	.long	23308
	.long	23309
	.long	23310
	.long	23311
.LCPI2_17:
	.long	25088
	.long	25089
	.long	25090
	.long	25091
	.long	25092
	.long	25093
	.long	25094
	.long	25095
	.long	25096
	.long	25097
	.long	25098
	.long	25099
	.long	25100
	.long	25101
	.long	25102
	.long	25103
.LCPI2_18:
	.long	26880
	.long	26881
	.long	26882
	.long	26883
	.long	26884
	.long	26885
	.long	26886
	.long	26887
	.long	26888
	.long	26889
	.long	26890
	.long	26891
	.long	26892
	.long	26893
	.long	26894
	.long	26895
.LCPI2_19:
	.long	28672
	.long	28673
	.long	28674
	.long	28675
	.long	28676
	.long	28677
	.long	28678
	.long	28679
	.long	28680
	.long	28681
	.long	28682
	.long	28683
	.long	28684
	.long	28685
	.long	28686
	.long	28687
.LCPI2_20:
	.long	30464
	.long	30465
	.long	30466
	.long	30467
	.long	30468
	.long	30469
	.long	30470
	.long	30471
	.long	30472
	.long	30473
	.long	30474
	.long	30475
	.long	30476
	.long	30477
	.long	30478
	.long	30479
.LCPI2_21:
	.long	32256
	.long	32257
	.long	32258
	.long	32259
	.long	32260
	.long	32261
	.long	32262
	.long	32263
	.long	32264
	.long	32265
	.long	32266
	.long	32267
	.long	32268
	.long	32269
	.long	32270
	.long	32271
.LCPI2_22:
	.long	34048
	.long	34049
	.long	34050
	.long	34051
	.long	34052
	.long	34053
	.long	34054
	.long	34055
	.long	34056
	.long	34057
	.long	34058
	.long	34059
	.long	34060
	.long	34061
	.long	34062
	.long	34063
.LCPI2_23:
	.long	35840
	.long	35841
	.long	35842
	.long	35843
	.long	35844
	.long	35845
	.long	35846
	.long	35847
	.long	35848
	.long	35849
	.long	35850
	.long	35851
	.long	35852
	.long	35853
	.long	35854
	.long	35855
.LCPI2_24:
	.long	37632
	.long	37633
	.long	37634
	.long	37635
	.long	37636
	.long	37637
	.long	37638
	.long	37639
	.long	37640
	.long	37641
	.long	37642
	.long	37643
	.long	37644
	.long	37645
	.long	37646
	.long	37647
.LCPI2_25:
	.long	39424
	.long	39425
	.long	39426
	.long	39427
	.long	39428
	.long	39429
	.long	39430
	.long	39431
	.long	39432
	.long	39433
	.long	39434
	.long	39435
	.long	39436
	.long	39437
	.long	39438
	.long	39439
.LCPI2_26:
	.long	41216
	.long	41217
	.long	41218
	.long	41219
	.long	41220
	.long	41221
	.long	41222
	.long	41223
	.long	41224
	.long	41225
	.long	41226
	.long	41227
	.long	41228
	.long	41229
	.long	41230
	.long	41231
.LCPI2_27:
	.long	43008
	.long	43009
	.long	43010
	.long	43011
	.long	43012
	.long	43013
	.long	43014
	.long	43015
	.long	43016
	.long	43017
	.long	43018
	.long	43019
	.long	43020
	.long	43021
	.long	43022
	.long	43023
.LCPI2_28:
	.long	44800
	.long	44801
	.long	44802
	.long	44803
	.long	44804
	.long	44805
	.long	44806
	.long	44807
	.long	44808
	.long	44809
	.long	44810
	.long	44811
	.long	44812
	.long	44813
	.long	44814
	.long	44815
.LCPI2_29:
	.long	46592
	.long	46593
	.long	46594
	.long	46595
	.long	46596
	.long	46597
	.long	46598
	.long	46599
	.long	46600
	.long	46601
	.long	46602
	.long	46603
	.long	46604
	.long	46605
	.long	46606
	.long	46607
.LCPI2_30:
	.long	48384
	.long	48385
	.long	48386
	.long	48387
	.long	48388
	.long	48389
	.long	48390
	.long	48391
	.long	48392
	.long	48393
	.long	48394
	.long	48395
	.long	48396
	.long	48397
	.long	48398
	.long	48399
	.text
	.p2align	4, 0x90
	.type	.L__tvm_parallel_lambda,@function
.L__tvm_parallel_lambda:
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$344, %rsp
	movq	%rdx, %rcx
	movl	8(%rsi), %esi
	leal	31(%rsi), %eax
	cltd
	idivl	%esi
	leal	1(%rdi), %edx
	imull	%eax, %edx
	cmpl	$33, %edx
	movl	$32, %esi
	cmovgel	%esi, %edx
	imull	%edi, %eax
	cmpl	$33, %eax
	cmovll	%eax, %esi
	cmpl	%edx, %esi
	jge	.LBB2_19
	addl	$1, %esi
	movslq	%esi, %rbx
	addq	$-1, %rbx
	notl	%eax
	cmpl	$-34, %eax
	movl	$-33, %esi
	cmovgl	%eax, %esi
	shll	$8, %esi
	leal	(%rsi,%rsi,8), %eax
	movl	$-1552, %r9d
	subl	%eax, %r9d
	movl	$-1568, %r12d
	subl	%eax, %r12d
	movl	$-1584, %r10d
	subl	%eax, %r10d
	movl	$-1792, %r8d
	subl	%eax, %r8d
	movl	$-1808, %r14d
	subl	%eax, %r14d
	movl	$-1824, %r15d
	subl	%eax, %r15d
	movl	$-1840, %r13d
	subl	%eax, %r13d
	movl	$-2048, %esi
	subl	%eax, %esi
	movq	%rsi, -24(%rsp)
	movl	$-2064, %esi
	subl	%eax, %esi
	movq	%rsi, -32(%rsp)
	movl	$-2080, %esi
	subl	%eax, %esi
	movq	%rsi, -40(%rsp)
	movl	$-2096, %esi
	subl	%eax, %esi
	movq	%rsi, -48(%rsp)
	movl	$-2304, %esi
	subl	%eax, %esi
	movq	%rsi, -56(%rsp)
	movq	8(%rcx), %rbp
	movq	16(%rcx), %rax
	movq	(%rcx), %r11
	movslq	%edx, %rcx
	movq	%rcx, 80(%rsp)
	addq	$196992, %rax
	movq	%rax, 88(%rsp)
	vmovdqa64	.LCPI2_20(%rip), %zmm30
	vmovdqa64	.LCPI2_21(%rip), %zmm31
	vmovdqa64	.LCPI2_22(%rip), %zmm20
	vmovdqa64	.LCPI2_23(%rip), %zmm22
	vmovdqa64	.LCPI2_24(%rip), %zmm23
	vmovdqa64	.LCPI2_25(%rip), %zmm24
	vmovdqa64	.LCPI2_26(%rip), %zmm25
	vmovdqa64	.LCPI2_27(%rip), %zmm26
	vmovdqa64	.LCPI2_28(%rip), %zmm27
	vmovdqa64	.LCPI2_29(%rip), %zmm28
	vmovdqa64	.LCPI2_30(%rip), %zmm29
.LBB2_2:
	leaq	(%rbx,%rbx,2), %rax
	movq	%rax, 176(%rsp)
	xorl	%eax, %eax
.LBB2_3:
	movq	%r11, -128(%rsp)
	movq	-56(%rsp), %rcx
	leaq	(%rcx,%rax), %rdx
	movslq	%edx, %rdx
	vxorps	%xmm1, %xmm1, %xmm1
	vmovaps	%zmm1, (%r11,%rdx,4)
	vpbroadcastd	%edx, %ymm0
	vmovdqa	.LCPI2_0(%rip), %ymm2
	vpor	%ymm2, %ymm0, %ymm0
	vmovq	%xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vpextrq	$1, %xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vextracti128	$1, %ymm0, %xmm0
	vmovq	%xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vpextrq	$1, %xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vpbroadcastd	%edx, %xmm0
	vmovdqa	.LCPI2_1(%rip), %xmm3
	vpor	%xmm3, %xmm0, %xmm0
	vpextrq	$1, %xmm0, %rdx
	vmovq	%xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	movslq	%edx, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	sarq	$32, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	movq	-48(%rsp), %rcx
	leal	(%rcx,%rax), %edx
	movslq	%edx, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	movq	-40(%rsp), %rcx
	leal	(%rcx,%rax), %edx
	movslq	%edx, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	movq	-32(%rsp), %rcx
	leal	(%rcx,%rax), %edx
	movslq	%edx, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	movq	-24(%rsp), %rcx
	leaq	(%rcx,%rax), %rdx
	movslq	%edx, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	vpbroadcastd	%edx, %ymm0
	vpor	%ymm2, %ymm0, %ymm0
	vmovq	%xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vpextrq	$1, %xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vextracti128	$1, %ymm0, %xmm0
	vmovq	%xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vpextrq	$1, %xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vpbroadcastd	%edx, %xmm0
	vpor	%xmm3, %xmm0, %xmm0
	vpextrq	$1, %xmm0, %rdx
	vmovq	%xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	movslq	%edx, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	sarq	$32, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	leal	(%rax,%r13), %edx
	movslq	%edx, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	leal	(%r15,%rax), %edx
	movslq	%edx, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	leal	(%r14,%rax), %edx
	movslq	%edx, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	leaq	(%r8,%rax), %rdx
	movslq	%edx, %rdx
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdx,4)
	vpbroadcastd	%edx, %ymm0
	vpor	%ymm2, %ymm0, %ymm0
	vmovq	%xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vpextrq	$1, %xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vextracti128	$1, %ymm0, %xmm0
	vmovq	%xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	sarq	$32, %rsi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rsi,4)
	vpextrq	$1, %xmm0, %rsi
	movslq	%esi, %rdi
	movq	-128(%rsp), %rcx
	vmovaps	%zmm1, (%rcx,%rdi,4)
	movq	-128(%rsp), %r11
	sarq	$32, %rsi
	vmovaps	%zmm1, (%r11,%rsi,4)
	vpbroadcastd	%edx, %xmm0
	vpor	%xmm3, %xmm0, %xmm0
	vmovq	%xmm0, %rdx
	movslq	%edx, %rsi
	vmovaps	%zmm1, (%r11,%rsi,4)
	vpextrq	$1, %xmm0, %rsi
	sarq	$32, %rdx
	vmovaps	%zmm1, (%r11,%rdx,4)
	movslq	%esi, %rdx
	vmovaps	%zmm1, (%r11,%rdx,4)
	sarq	$32, %rsi
	vmovaps	%zmm1, (%r11,%rsi,4)
	leal	(%r10,%rax), %edx
	movslq	%edx, %rdx
	vmovaps	%zmm1, (%r11,%rdx,4)
	leal	(%r12,%rax), %edx
	movslq	%edx, %rdx
	vmovaps	%zmm1, (%r11,%rdx,4)
	leal	(%r9,%rax), %edx
	movslq	%edx, %rdx
	vmovaps	%zmm1, (%r11,%rdx,4)
	addq	$768, %rax
	cmpq	$2304, %rax
	jne	.LBB2_3
	.p2align	4, 0x90
	movq	%r13, 96(%rsp)
	movq	%r15, 104(%rsp)
	movq	%r14, 112(%rsp)
	movq	%r8, 120(%rsp)
	movq	%r10, 128(%rsp)
	movq	%r9, %r10
	movl	%ebx, %eax
	sarl	$2, %eax
	movl	%ebx, %edx
	sarl	$31, %edx
	shrl	$30, %edx
	addl	%ebx, %edx
	andl	$-4, %edx
	movq	%rbx, %r13
	movl	%ebx, %esi
	subl	%edx, %esi
	movslq	%esi, %rdx
	imulq	$831744, %rdx, %r15
	addq	88(%rsp), %r15
	imull	$200704, %eax, %eax
	leal	15(%rax), %r8d
	leal	14(%rax), %r14d
	movl	%eax, %ecx
	orl	$13, %ecx
	xorl	%r9d, %r9d
	movq	%r9, -64(%rsp)
.LBB2_5:
	movl	%ecx, -88(%rsp)
	movl	%ecx, %ebx
	movl	%r14d, %edx
	movl	%r8d, %esi
	movq	%rax, %r9
	movq	%r15, %rdi
	xorl	%ecx, %ecx
.LBB2_6:
	movq	%rcx, 40(%rsp)
	movl	%r14d, -84(%rsp)
	movl	%r8d, -80(%rsp)
	movq	%rax, 64(%rsp)
	movq	%r15, 72(%rsp)
	movq	%r12, 136(%rsp)
	movq	%r10, 144(%rsp)
	movq	%r13, 152(%rsp)
	movl	%ebx, -92(%rsp)
	movl	%ebx, %eax
	movl	%edx, -96(%rsp)
	movl	%edx, %ecx
	movl	%esi, -100(%rsp)
	movl	%esi, %edx
	movq	%r9, 56(%rsp)
	movq	%r9, %rsi
	movq	%rdi, 48(%rsp)
	movq	%rdi, %r10
	xorl	%ebx, %ebx
.LBB2_7:
	movq	%rbx, 16(%rsp)
	movl	%eax, -104(%rsp)
	movl	%eax, %ebx
	movl	%ecx, -108(%rsp)
	movl	%edx, -112(%rsp)
	movl	%edx, %eax
	movq	%rsi, 32(%rsp)
	movq	%r10, 24(%rsp)
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB2_8:
	movq	%rdx, 160(%rsp)
	movl	%eax, -76(%rsp)
	cltq
	leaq	(,%rax,4), %rax
	addq	%rbp, %rax
	movq	%rax, 216(%rsp)
	movl	%ecx, -72(%rsp)
	movslq	%ecx, %rdx
	leaq	(%rbp,%rdx,4), %rax
	movq	%rax, 336(%rsp)
	movl	%ebx, -68(%rsp)
	movslq	%ebx, %rdx
	leaq	(%rbp,%rdx,4), %rax
	movq	%rax, 208(%rsp)
	movq	%rsi, 8(%rsp)
	movslq	%esi, %rdx
	leaq	(,%rdx,4), %rax
	addq	%rbp, %rax
	movq	%rax, 200(%rsp)
	movq	%r10, 168(%rsp)
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB2_9:
	movq	176(%rsp), %rax
	movq	%rdx, 184(%rsp)
	leal	(%rdx,%rax), %edx
	leal	(%rdx,%rdx,2), %eax
	movq	%rax, 224(%rsp)
	movq	%r10, 192(%rsp)
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB2_10:
	movq	224(%rsp), %rax
	movq	%rcx, 328(%rsp)
	leal	(%rcx,%rax), %edx
	shll	$8, %edx
	movslq	%edx, %rax
	vpbroadcastd	%eax, %ymm0
	vpor	.LCPI2_0(%rip), %ymm0, %ymm0
	vmovq	%xmm0, %r15
	movslq	%r15d, %r8
	sarq	$32, %r15
	vpextrq	$1, %xmm0, %rbx
	movslq	%ebx, %rcx
	movq	%rcx, (%rsp)
	sarq	$32, %rbx
	vextracti128	$1, %ymm0, %xmm0
	vmovq	%xmm0, %rsi
	movslq	%esi, %rcx
	movq	%rcx, -8(%rsp)
	sarq	$32, %rsi
	vpextrq	$1, %xmm0, %rdi
	movslq	%edi, %rcx
	movq	%rcx, -16(%rsp)
	sarq	$32, %rdi
	vpbroadcastd	%eax, %xmm0
	vpor	.LCPI2_1(%rip), %xmm0, %xmm0
	vpextrq	$1, %xmm0, %rdx
	movslq	%edx, %r13
	movq	%rdx, %rcx
	sarq	$32, %rcx
	vmovq	%xmm0, %r12
	movslq	%r12d, %r9
	sarq	$32, %r12
	leal	208(%rax), %edx
	movslq	%edx, %r14
	leal	224(%rax), %edx
	movl	%edx, -120(%rsp)
	vmovaps	(%r11,%rax,4), %zmm17
	movq	%r8, 312(%rsp)
	vmovaps	(%r11,%r8,4), %zmm16
	movq	%r15, 320(%rsp)
	vmovaps	(%r11,%r15,4), %zmm15
	movq	(%rsp), %rdx
	vmovaps	(%r11,%rdx,4), %zmm14
	movq	%rbx, 304(%rsp)
	vmovaps	(%r11,%rbx,4), %zmm13
	movq	-8(%rsp), %rdx
	vmovaps	(%r11,%rdx,4), %zmm12
	movq	%rsi, 296(%rsp)
	vmovaps	(%r11,%rsi,4), %zmm11
	movq	-16(%rsp), %rdx
	vmovaps	(%r11,%rdx,4), %zmm10
	movq	%rdi, 288(%rsp)
	vmovaps	(%r11,%rdi,4), %zmm9
	movq	%r9, 248(%rsp)
	vmovaps	(%r11,%r9,4), %zmm8
	movq	%r12, 256(%rsp)
	vmovaps	(%r11,%r12,4), %zmm7
	movq	%r13, 272(%rsp)
	vmovaps	(%r11,%r13,4), %zmm6
	movslq	-120(%rsp), %rsi
	movq	%rcx, 280(%rsp)
	vmovaps	(%r11,%rcx,4), %zmm5
	movq	%r14, 264(%rsp)
	vmovaps	(%r11,%r14,4), %zmm4
	movq	%rax, -120(%rsp)
	leal	240(%rax), %edx
	movslq	%edx, %rax
	movq	%rsi, 240(%rsp)
	vmovaps	(%r11,%rsi,4), %zmm3
	movq	%rax, 232(%rsp)
	vmovaps	(%r11,%rax,4), %zmm0
	movq	$-896, %r8
	movq	216(%rsp), %r11
	movq	208(%rsp), %r15
	movq	200(%rsp), %r9
	.p2align	4, 0x90
.LBB2_11:
	movq	8(%rsp), %rax
	leal	(%rax,%r8), %esi
	addl	$896, %esi
	vmovaps	-193408(%r10,%r8,4), %zmm18
	vfmadd231ps	3584(%r9,%r8,4){1to16}, %zmm18, %zmm17
	vpbroadcastd	%esi, %ymm19
	vporq	.LCPI2_2(%rip), %ymm19, %ymm19
	vmovq	%xmm19, %rdx
	movslq	%edx, %rdi
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm18, %zmm16
	sarq	$32, %rdx
	vpextrq	$1, %xmm19, %rdi
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm18, %zmm15
	movslq	%edi, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm18, %zmm14
	sarq	$32, %rdi
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm18, %zmm13
	vextracti32x4	$1, %ymm19, %xmm1
	vmovq	%xmm1, %rdx
	movslq	%edx, %rdi
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm18, %zmm12
	sarq	$32, %rdx
	vpextrq	$1, %xmm1, %rdi
	vpbroadcastd	%esi, %xmm1
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm18, %zmm11
	vpor	.LCPI2_3(%rip), %xmm1, %xmm1
	vpextrq	$1, %xmm1, %rdx
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rax
	movslq	%edi, %rbx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm18, %zmm8
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm18, %zmm7
	sarq	$32, %rdi
	movslq	%edx, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm18, %zmm6
	sarq	$32, %rdx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm18, %zmm10
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm18, %zmm5
	vfmadd231ps	3584(%r15,%r8,4){1to16}, %zmm18, %zmm4
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm18, %zmm9
	movq	336(%rsp), %rax
	vfmadd231ps	3584(%rax,%r8,4){1to16}, %zmm18, %zmm3
	vfmadd231ps	3584(%r11,%r8,4){1to16}, %zmm18, %zmm0
	vpbroadcastd	%esi, %zmm18
	vpaddd	.LCPI2_4(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-186112(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti32x4	$1, %ymm1, %xmm21
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm21, %rdi
	vmovq	%xmm21, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_5(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-178816(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_6(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-171520(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_7(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-164224(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_8(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-156928(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_9(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-149632(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_10(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-142336(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_11(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-135040(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_12(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-127744(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_13(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-120448(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_14(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-113152(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_15(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-105856(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_16(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-98560(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_17(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-91264(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_18(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-83968(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	.LCPI2_19(%rip), %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-76672(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm30, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-69376(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm31, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-62080(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm20, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-54784(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm22, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-47488(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm23, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-40192(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm24, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-32896(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm25, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-25600(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm26, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-18304(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm27, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %rax
	movslq	%eax, %rcx
	vmovaps	-11008(%r10,%r8,4), %zmm19
	sarq	$32, %rax
	movslq	%esi, %r14
	sarq	$32, %rsi
	vextracti128	$1, %ymm1, %xmm2
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm17
	vpextrq	$1, %xmm2, %rdi
	vmovq	%xmm2, %rdx
	movslq	%edx, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm16
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm2, %rbx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm15
	movslq	%ebx, %r14
	sarq	$32, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdx
	movslq	%eax, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm13
	movslq	%edi, %rcx
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm12
	sarq	$32, %rdi
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rax
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm11
	vmovq	%xmm1, %rcx
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm4
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm9
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm3
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm28, %zmm18, %zmm1
	vpextrq	$1, %xmm1, %r14
	vmovq	%xmm1, %rdx
	movslq	%edx, %rax
	vmovaps	-3712(%r10,%r8,4), %zmm19
	sarq	$32, %rdx
	vextracti128	$1, %ymm1, %xmm2
	vpextrq	$1, %xmm2, %rsi
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm17
	vmovq	%xmm2, %rdi
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm16
	vmovq	%xmm2, %rcx
	movslq	%r14d, %rdx
	sarq	$32, %r14
	movslq	%edi, %rbx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm15
	movslq	%ecx, %rdx
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm8
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm19, %zmm14
	sarq	$32, %rdi
	movslq	%eax, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm7
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm13
	sarq	$32, %rax
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rcx
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm6
	vmovq	%xmm1, %rax
	movslq	%eax, %rbx
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm19, %zmm5
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm12
	movslq	%esi, %rdi
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm4
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm19, %zmm11
	sarq	$32, %rsi
	movslq	%ecx, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm19, %zmm3
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm19, %zmm10
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm19, %zmm0
	vpaddd	%zmm29, %zmm18, %zmm1
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm19, %zmm9
	vextracti32x4	$1, %ymm1, %xmm18
	vpextrq	$1, %xmm18, %r14
	vextracti32x4	$2, %zmm1, %xmm2
	vpextrq	$1, %xmm2, %rax
	vmovq	%xmm18, %rcx
	vmovq	%xmm2, %rdx
	vpextrq	$1, %xmm1, %rdi
	vmovq	%xmm1, %rbx
	vextracti32x4	$3, %zmm1, %xmm1
	vpextrq	$1, %xmm1, %rsi
	vmovq	%xmm1, %r13
	vmovaps	3584(%r10,%r8,4), %zmm1
	movslq	%ebx, %r12
	vfmadd231ps	(%rbp,%r12,4){1to16}, %zmm1, %zmm17
	sarq	$32, %rbx
	movslq	%edi, %r12
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm1, %zmm16
	vfmadd231ps	(%rbp,%r12,4){1to16}, %zmm1, %zmm15
	sarq	$32, %rdi
	movslq	%edx, %rbx
	vfmadd231ps	(%rbp,%rdi,4){1to16}, %zmm1, %zmm14
	sarq	$32, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm1, %zmm8
	movslq	%ecx, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm1, %zmm13
	movslq	%eax, %rdx
	vfmadd231ps	(%rbp,%rdx,4){1to16}, %zmm1, %zmm7
	sarq	$32, %rcx
	vfmadd231ps	(%rbp,%rcx,4){1to16}, %zmm1, %zmm12
	sarq	$32, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm1, %zmm6
	movslq	%r14d, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm1, %zmm11
	movslq	%r13d, %rax
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm1, %zmm5
	sarq	$32, %r14
	vfmadd231ps	(%rbp,%r14,4){1to16}, %zmm1, %zmm10
	sarq	$32, %r13
	vfmadd231ps	(%rbp,%r13,4){1to16}, %zmm1, %zmm4
	movslq	%esi, %rax
	vfmadd231ps	(%rbp,%rbx,4){1to16}, %zmm1, %zmm9
	vfmadd231ps	(%rbp,%rax,4){1to16}, %zmm1, %zmm3
	sarq	$32, %rsi
	vfmadd231ps	(%rbp,%rsi,4){1to16}, %zmm1, %zmm0
	addq	$16, %r8
	jne	.LBB2_11
	movq	-128(%rsp), %r11
	movq	-120(%rsp), %rax
	vmovaps	%zmm17, (%r11,%rax,4)
	movq	312(%rsp), %rax
	vmovaps	%zmm16, (%r11,%rax,4)
	movq	320(%rsp), %rax
	vmovaps	%zmm15, (%r11,%rax,4)
	movq	(%rsp), %rax
	vmovaps	%zmm14, (%r11,%rax,4)
	movq	304(%rsp), %rax
	vmovaps	%zmm13, (%r11,%rax,4)
	movq	-8(%rsp), %rax
	vmovaps	%zmm12, (%r11,%rax,4)
	movq	296(%rsp), %rax
	vmovaps	%zmm11, (%r11,%rax,4)
	movq	-16(%rsp), %rax
	vmovaps	%zmm10, (%r11,%rax,4)
	movq	288(%rsp), %rax
	vmovaps	%zmm9, (%r11,%rax,4)
	movq	248(%rsp), %rax
	vmovaps	%zmm8, (%r11,%rax,4)
	movq	256(%rsp), %rax
	vmovaps	%zmm7, (%r11,%rax,4)
	movq	272(%rsp), %rax
	vmovaps	%zmm6, (%r11,%rax,4)
	movq	280(%rsp), %rax
	vmovaps	%zmm5, (%r11,%rax,4)
	movq	264(%rsp), %rax
	vmovaps	%zmm4, (%r11,%rax,4)
	movq	240(%rsp), %rax
	vmovaps	%zmm3, (%r11,%rax,4)
	movq	232(%rsp), %rax
	vmovaps	%zmm0, (%r11,%rax,4)
	movq	328(%rsp), %rcx
	addq	$1, %rcx
	addq	$64, %r10
	cmpq	$3, %rcx
	jne	.LBB2_10
	movq	184(%rsp), %rdx
	addq	$1, %rdx
	movq	192(%rsp), %r10
	addq	$7296, %r10
	cmpq	$3, %rdx
	jne	.LBB2_9
	movq	160(%rsp), %rdx
	addq	$1, %rdx
	movq	168(%rsp), %r10
	addq	$204288, %r10
	movq	8(%rsp), %rsi
	addq	$50176, %rsi
	movl	-76(%rsp), %eax
	addl	$50176, %eax
	movl	-72(%rsp), %ecx
	addl	$50176, %ecx
	movl	-68(%rsp), %ebx
	addl	$50176, %ebx
	cmpq	$2, %rdx
	jne	.LBB2_8
	.p2align	4, 0x90
	movq	16(%rsp), %rbx
	addq	$1, %rbx
	movq	24(%rsp), %r10
	addq	$3584, %r10
	movq	32(%rsp), %rsi
	addq	$896, %rsi
	movl	-112(%rsp), %edx
	addl	$896, %edx
	movl	-108(%rsp), %ecx
	addl	$896, %ecx
	movl	-104(%rsp), %eax
	addl	$896, %eax
	cmpq	$2, %rbx
	jne	.LBB2_7
	.p2align	4, 0x90
	movq	40(%rsp), %rcx
	addq	$1, %rcx
	movq	48(%rsp), %rdi
	addq	$408576, %rdi
	movq	56(%rsp), %r9
	addq	$100352, %r9
	movl	-100(%rsp), %esi
	addl	$100352, %esi
	movl	-96(%rsp), %edx
	addl	$100352, %edx
	movl	-92(%rsp), %ebx
	addl	$100352, %ebx
	cmpq	$2, %rcx
	movq	152(%rsp), %r13
	movq	144(%rsp), %r10
	movq	136(%rsp), %r12
	movq	72(%rsp), %r15
	movq	64(%rsp), %rax
	movl	-80(%rsp), %r8d
	movl	-84(%rsp), %r14d
	jne	.LBB2_6
	.p2align	4, 0x90
	movq	-64(%rsp), %rbx
	movq	%rbx, %r9
	addq	$1, %r9
	addq	$3326976, %r15
	addq	$1605632, %rax
	addl	$1605632, %r8d
	addl	$1605632, %r14d
	movl	-88(%rsp), %ecx
	addl	$1605632, %ecx
	movq	%r9, %rbx
	movq	%rbx, -64(%rsp)
	cmpq	$64, %r9
	jne	.LBB2_5
	.p2align	4, 0x90
	movq	%r13, %rbx
	addq	$1, %rbx
	movq	%r10, %r9
	addq	$2304, %r9
	addq	$2304, %r12
	movq	128(%rsp), %r10
	addq	$2304, %r10
	movq	120(%rsp), %r8
	addq	$2304, %r8
	movq	112(%rsp), %r14
	addq	$2304, %r14
	movq	104(%rsp), %r15
	addq	$2304, %r15
	movq	96(%rsp), %r13
	addq	$2304, %r13
	addq	$2304, -24(%rsp)
	addq	$2304, -32(%rsp)
	addq	$2304, -40(%rsp)
	addq	$2304, -48(%rsp)
	addq	$2304, -56(%rsp)
	cmpq	80(%rsp), %rbx
	jl	.LBB2_2
.LBB2_19:
	xorl	%eax, %eax
	addq	$344, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end2:
	.size	.L__tvm_parallel_lambda, .Lfunc_end2-.L__tvm_parallel_lambda

	.type	__TVMAPISetLastError,@object
	.bss
	.weak	__TVMAPISetLastError
	.p2align	3
__TVMAPISetLastError:
	.quad	0
	.size	__TVMAPISetLastError, 8

	.type	__TVMBackendParallelLaunch,@object
	.weak	__TVMBackendParallelLaunch
	.p2align	3
__TVMBackendParallelLaunch:
	.quad	0
	.size	__TVMBackendParallelLaunch, 8

	.type	.L.str,@object
	.section	.rodata,"a",@progbits
.L.str:
	.asciz	"Assert fail: (num_args == 3), default_function: num_args should be 3"
	.size	.L.str, 69

	.type	.L.str.1,@object
.L.str.1:
	.asciz	"Assert fail: (((((1 == int32(arg0.strides[4])) && ((1*16) == int32(arg0.strides[3]))) && (((1*16)*112) == int32(arg0.strides[2]))) && ((((1*16)*112)*112) == int32(arg0.strides[1]))) && (((((1*16)*112)*112)*8) == int32(arg0.strides[0]))), arg0.strides: expected to be compact array"
	.size	.L.str.1, 281

	.type	.L.str.2,@object
.L.str.2:
	.asciz	"Assert fail: (((((1 == int32(arg1.strides[4])) && ((1*16) == int32(arg1.strides[3]))) && (((1*16)*114) == int32(arg1.strides[2]))) && ((((1*16)*114)*114) == int32(arg1.strides[1]))) && (((((1*16)*114)*114)*4) == int32(arg1.strides[0]))), arg1.strides: expected to be compact array"
	.size	.L.str.2, 281

	.type	.L.str.3,@object
.L.str.3:
	.asciz	"Assert fail: ((((((1 == int32(arg2.strides[5])) && ((1*16) == int32(arg2.strides[4]))) && (((1*16)*16) == int32(arg2.strides[3]))) && ((((1*16)*16)*3) == int32(arg2.strides[2]))) && (((((1*16)*16)*3)*3) == int32(arg2.strides[1]))) && ((((((1*16)*16)*3)*3)*4) == int32(arg2.strides[0]))), arg2.strides: expected to be compact array"
	.size	.L.str.3, 331

	.type	.L.str.4,@object
.L.str.4:
	.asciz	"Assert fail: (((arg0.code == 3) || (arg0.code == 7)) || (arg0.code == 4)), default_function: Expect arg[0] to be pointer"
	.size	.L.str.4, 121

	.type	.L.str.5,@object
.L.str.5:
	.asciz	"Assert fail: (((arg1.code == 3) || (arg1.code == 7)) || (arg1.code == 4)), default_function: Expect arg[1] to be pointer"
	.size	.L.str.5, 121

	.type	.L.str.6,@object
.L.str.6:
	.asciz	"Assert fail: (((arg2.code == 3) || (arg2.code == 7)) || (arg2.code == 4)), default_function: Expect arg[2] to be pointer"
	.size	.L.str.6, 121

	.type	.L.str.7,@object
.L.str.7:
	.asciz	"Assert fail: (dev_type == 1), device_type need to be 1"
	.size	.L.str.7, 55

	.type	.L.str.8,@object
.L.str.8:
	.asciz	"Assert fail: (5 == tvm_struct_get(arg0, 0, 4)), arg0.ndim is expected to equal 5"
	.size	.L.str.8, 81

	.type	.L.str.9,@object
.L.str.9:
	.asciz	"Assert fail: (((tvm_struct_get(arg0, 0, 5) == (uint8)2) && (tvm_struct_get(arg0, 0, 6) == (uint8)32)) && (tvm_struct_get(arg0, 0, 7) == (uint16)1)), arg0.dtype is expected to be float32"
	.size	.L.str.9, 186

	.type	.L.str.10,@object
.L.str.10:
	.asciz	"Assert fail: (int32(arg0.shape[0]) == 64), Argument arg0.shape[0] has an unsatisfied constraint"
	.size	.L.str.10, 96

	.type	.L.str.11,@object
.L.str.11:
	.asciz	"Assert fail: (int32(arg0.shape[1]) == 8), Argument arg0.shape[1] has an unsatisfied constraint"
	.size	.L.str.11, 95

	.type	.L.str.12,@object
.L.str.12:
	.asciz	"Assert fail: (int32(arg0.shape[2]) == 112), Argument arg0.shape[2] has an unsatisfied constraint"
	.size	.L.str.12, 97

	.type	.L.str.13,@object
.L.str.13:
	.asciz	"Assert fail: (int32(arg0.shape[3]) == 112), Argument arg0.shape[3] has an unsatisfied constraint"
	.size	.L.str.13, 97

	.type	.L.str.14,@object
.L.str.14:
	.asciz	"Assert fail: (int32(arg0.shape[4]) == 16), Argument arg0.shape[4] has an unsatisfied constraint"
	.size	.L.str.14, 96

	.type	.L.str.15,@object
.L.str.15:
	.asciz	"Assert fail: (tvm_struct_get(arg0, 0, 8) == (uint64)0), Argument arg0.byte_offset has an unsatisfied constraint"
	.size	.L.str.15, 112

	.type	.L.str.16,@object
.L.str.16:
	.asciz	"Assert fail: (5 == tvm_struct_get(arg1, 0, 4)), arg1.ndim is expected to equal 5"
	.size	.L.str.16, 81

	.type	.L.str.17,@object
.L.str.17:
	.asciz	"Assert fail: (((tvm_struct_get(arg1, 0, 5) == (uint8)2) && (tvm_struct_get(arg1, 0, 6) == (uint8)32)) && (tvm_struct_get(arg1, 0, 7) == (uint16)1)), arg1.dtype is expected to be float32"
	.size	.L.str.17, 186

	.type	.L.str.18,@object
.L.str.18:
	.asciz	"Assert fail: (int32(arg1.shape[0]) == 64), Argument arg1.shape[0] has an unsatisfied constraint"
	.size	.L.str.18, 96

	.type	.L.str.19,@object
.L.str.19:
	.asciz	"Assert fail: (int32(arg1.shape[1]) == 4), Argument arg1.shape[1] has an unsatisfied constraint"
	.size	.L.str.19, 95

	.type	.L.str.20,@object
.L.str.20:
	.asciz	"Assert fail: (int32(arg1.shape[2]) == 114), Argument arg1.shape[2] has an unsatisfied constraint"
	.size	.L.str.20, 97

	.type	.L.str.21,@object
.L.str.21:
	.asciz	"Assert fail: (int32(arg1.shape[3]) == 114), Argument arg1.shape[3] has an unsatisfied constraint"
	.size	.L.str.21, 97

	.type	.L.str.22,@object
.L.str.22:
	.asciz	"Assert fail: (int32(arg1.shape[4]) == 16), Argument arg1.shape[4] has an unsatisfied constraint"
	.size	.L.str.22, 96

	.type	.L.str.23,@object
.L.str.23:
	.asciz	"Assert fail: (tvm_struct_get(arg1, 0, 8) == (uint64)0), Argument arg1.byte_offset has an unsatisfied constraint"
	.size	.L.str.23, 112

	.type	.L.str.24,@object
.L.str.24:
	.asciz	"Assert fail: (1 == tvm_struct_get(arg1, 0, 10)), Argument arg1.device_type has an unsatisfied constraint"
	.size	.L.str.24, 105

	.type	.L.str.25,@object
.L.str.25:
	.asciz	"Assert fail: (dev_id == tvm_struct_get(arg1, 0, 9)), Argument arg1.device_id has an unsatisfied constraint"
	.size	.L.str.25, 107

	.type	.L.str.26,@object
.L.str.26:
	.asciz	"Assert fail: (6 == tvm_struct_get(arg2, 0, 4)), arg2.ndim is expected to equal 6"
	.size	.L.str.26, 81

	.type	.L.str.27,@object
.L.str.27:
	.asciz	"Assert fail: (((tvm_struct_get(arg2, 0, 5) == (uint8)2) && (tvm_struct_get(arg2, 0, 6) == (uint8)32)) && (tvm_struct_get(arg2, 0, 7) == (uint16)1)), arg2.dtype is expected to be float32"
	.size	.L.str.27, 186

	.type	.L.str.28,@object
.L.str.28:
	.asciz	"Assert fail: (int32(arg2.shape[0]) == 8), Argument arg2.shape[0] has an unsatisfied constraint"
	.size	.L.str.28, 95

	.type	.L.str.29,@object
.L.str.29:
	.asciz	"Assert fail: (int32(arg2.shape[1]) == 4), Argument arg2.shape[1] has an unsatisfied constraint"
	.size	.L.str.29, 95

	.type	.L.str.30,@object
.L.str.30:
	.asciz	"Assert fail: (int32(arg2.shape[2]) == 3), Argument arg2.shape[2] has an unsatisfied constraint"
	.size	.L.str.30, 95

	.type	.L.str.31,@object
.L.str.31:
	.asciz	"Assert fail: (int32(arg2.shape[3]) == 3), Argument arg2.shape[3] has an unsatisfied constraint"
	.size	.L.str.31, 95

	.type	.L.str.32,@object
.L.str.32:
	.asciz	"Assert fail: (int32(arg2.shape[4]) == 16), Argument arg2.shape[4] has an unsatisfied constraint"
	.size	.L.str.32, 96

	.type	.L.str.33,@object
.L.str.33:
	.asciz	"Assert fail: (int32(arg2.shape[5]) == 16), Argument arg2.shape[5] has an unsatisfied constraint"
	.size	.L.str.33, 96

	.type	.L.str.34,@object
.L.str.34:
	.asciz	"Assert fail: (tvm_struct_get(arg2, 0, 8) == (uint64)0), Argument arg2.byte_offset has an unsatisfied constraint"
	.size	.L.str.34, 112

	.type	.L.str.35,@object
.L.str.35:
	.asciz	"Assert fail: (1 == tvm_struct_get(arg2, 0, 10)), Argument arg2.device_type has an unsatisfied constraint"
	.size	.L.str.35, 105

	.type	.L.str.36,@object
.L.str.36:
	.asciz	"Assert fail: (dev_id == tvm_struct_get(arg2, 0, 9)), Argument arg2.device_id has an unsatisfied constraint"
	.size	.L.str.36, 107

	.type	__tvm_main__,@object
	.weak	__tvm_main__
__tvm_main__:
	.asciz	"default_function"
	.size	__tvm_main__, 17


	.section	".note.GNU-stack","",@progbits
