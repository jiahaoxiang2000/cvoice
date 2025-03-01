from pathlib import Path
import pytest
import os
from unittest.mock import MagicMock, patch
from ..text.improver import TextImprover


@pytest.fixture
def sample_srt_content():
    return """
00:00:00 --> 00:00:06
今天是2025的1月7号

00:00:06 --> 00:00:09
我们之后主要是做一个论文的阅读

00:00:09 --> 00:00:18
我们主要是看一个SBH NCS的后浪解写名算法

00:00:18 --> 00:00:25
对应我们论文的一部分内容进行一部分的修改

00:00:25 --> 00:00:30
主要是分化一下实现的一个算法

00:00:30 --> 00:00:32
学长也都知道

00:00:32 --> 00:00:38
我们去关注一下后浪子密码标准的论文工作

00:00:38 --> 00:00:40
在他的PQC项目下

00:00:40 --> 00:00:41
在PQC项目下

00:00:41 --> 00:00:44
他在今年的8月13号

00:00:44 --> 00:00:47
他发布了最终的一个确定了最终的一个标准

00:00:47 --> 00:00:53
其中SBH NCS它就是其中一个实用的算法之一

00:00:53 --> 00:00:55
它是一个无照相的算法

00:00:55 --> 00:00:57
它是一个还是签名的一个方案

00:00:57 --> 00:00:59
它的一个特点就是

00:00:59 --> 00:01:03
它相较于它的一个数字签名

00:01:03 --> 00:01:06
它能够去抵抗一个样子介绍的一个问题

00:01:06 --> 00:01:10
我们SBH在它的

00:01:10 --> 00:01:12
它提交的第三个

00:01:12 --> 00:01:16
第三个提交的一个文档上

00:01:16 --> 00:01:18
提交的一个方案上面

00:01:18 --> 00:01:23
我们对它做一个展开了一个优化实现工作

00:01:23 --> 00:01:25
那么关于这个方案

00:01:25 --> 00:01:27
就是SBH NCS这个算法

00:01:27 --> 00:01:29
它主要有三个部分

00:01:29 --> 00:01:32
因为主要是

00:01:32 --> 00:01:33
这个过程主要是对

00:01:33 --> 00:01:35
这个消息先做一个签名

00:01:35 --> 00:01:37
然后再做一个HT的签名

00:01:37 --> 00:01:40
就是这个FOSRS的一个签名

00:01:40 --> 00:01:42
就是HT的签名

00:01:42 --> 00:01:43
然后这两个签名都会生成

00:01:43 --> 00:01:46
签字到一个true的一个生成

00:01:46 --> 00:01:54
然后再我们看这个表一表一是一篇

00:01:54 --> 00:01:56
全材货币的一个文章

00:01:56 --> 00:02:01
一篇读文上演示去做实现的一个优化

00:02:01 --> 00:02:04
这个算法的一篇文章

00:02:04 --> 00:02:06
其实会有三个部分

00:02:06 --> 00:02:08
就是上面说的是

00:02:08 --> 00:02:12
做按签名的这三个部分所占用的一个时间

00:02:12 --> 00:02:14
还有一个就是BIF

00:02:14 --> 00:02:18
关于它的个维持因素的一个生成

00:02:18 --> 00:02:22
现在这上面也占用了很长的一个时间

00:02:22 --> 00:02:23
所以我们也很感谢它

00:02:23 --> 00:02:25
占用了很长的一个时间

00:02:25 --> 00:02:31
那么其实我们就可以从这个三个方面去考虑

00:02:31 --> 00:02:32
这四个方面

00:02:32 --> 00:02:34
一个是HT的FOSRS的

00:02:34 --> 00:02:36
然后还有IC的

00:02:36 --> 00:02:38
IC的一个消息的

00:02:38 --> 00:02:40
以及说这个维持因素

00:02:40 --> 00:02:42
四个方面我们去

00:02:42 --> 00:02:46
把一个一个备优的一个实验方案

00:02:46 --> 00:02:49
那么写字方面我们就是把题目这个

00:02:49 --> 00:02:51
把它分拆一下

00:02:51 --> 00:02:52
然后

00:02:52 --> 00:02:53
然后在老师账页里面

00:02:53 --> 00:02:54
我们就添加了这个

00:02:54 --> 00:02:56
后量比算法这个

00:02:56 --> 00:02:58
这个

00:02:58 --> 00:03:02
然后我们把比占用的一个工具的一些分析

00:03:02 --> 00:03:04
然后里面的话我们

00:03:04 --> 00:03:05
我们就要去阐述

00:03:05 --> 00:03:06
我们把量子的程序

00:03:06 --> 00:03:08
现有的一个威胁

00:03:08 --> 00:03:11
然后这个SBHCS在

00:03:11 --> 00:03:14
这个后量比命法中它有一个

00:03:14 --> 00:03:15
它是一个设计的算法

00:03:15 --> 00:03:17
所以它是比较重要的一个算法

00:03:17 --> 00:03:18
但是同时呢

00:03:18 --> 00:03:20
它这个算法的

00:03:20 --> 00:03:22
即使开销相较于传统的一个链接运方案

00:03:22 --> 00:03:24
就是比较大

00:03:24 --> 00:03:25
那么的话

00:03:25 --> 00:03:27
这就需要我们去

00:03:27 --> 00:03:30
对它进行一个更高效的一个实现了

00:03:30 --> 00:03:32
我们就提出这个用

00:03:32 --> 00:03:35
GPU等地形设计上去加速这个链接运

00:03:35 --> 00:03:36
甚至每个部门

00:03:36 --> 00:03:38
一个大层主的

00:03:38 --> 00:03:40
一个情况下

00:03:40 --> 00:03:42
我们这个参考文献

00:03:42 --> 00:03:43
它这个

00:03:43 --> 00:03:48
SBHCS这个算法呢是在

00:03:48 --> 00:03:51
2019年被提出的

00:03:51 --> 00:03:55
然后我们也有分析一个研究

00:03:55 --> 00:03:58
是对它做一个

00:03:58 --> 00:03:59
有关实现的

00:03:59 --> 00:04:04
那么以上就是我们的工作

00:04:04 --> 00:04:06
这一周的工作

00:04:06 --> 00:04:09
我们修中想开一个新的系列

00:04:09 --> 00:04:10
就是Embedded系列

00:04:10 --> 00:04:11
对不对

00:04:11 --> 00:04:12
Embedded系列

00:04:12 --> 00:04:15
也期待就是说

00:04:15 --> 00:04:18
我们之前也做了几个系列

00:04:18 --> 00:04:19
做了一点

00:04:19 --> 00:04:20
第一第一第一

00:04:20 --> 00:04:21
第一第一第一

00:04:21 --> 00:04:22
第一第一第一

00:04:22 --> 00:04:23
做一段时间

00:04:23 --> 00:04:24
但是没有一段时间

00:04:24 --> 00:04:25
所以我们就

00:04:25 --> 00:04:27
接着出这个视频吧

00:04:27 --> 00:04:28
去讲一下这个

00:04:28 --> 00:04:28
我们的工作

"""

@pytest.fixture
def improver():
    return TextImprover()

def test_parse_srt(improver, sample_srt_content):
    segments = improver._parse_srt(sample_srt_content)
    print(segments)
    assert len(segments) == 106
    assert segments[0]["text"] == "今天是2025的1月7号"
    assert segments[0]["timeline"] == "00:00:00 --> 00:00:06"

def test_estimate_tokens(improver):
    text = "This is a test sentence."
    tokens = improver._estimate_tokens(text)
    assert tokens == len(text) // 4

def test_batch_segments(improver):
    segments = [
        {"text": "a" * 1000},  # ~250 tokens
        {"text": "b" * 1000},  # ~250 tokens
        {"text": "c" * 8000},  # ~2000 tokens
    ]
    batches = improver._batch_segments(segments)
    assert len(batches) > 1  # Should split into multiple batches

@pytest.mark.asyncio
async def test_improve_text(improver, sample_srt_content, temp_dir):
    # Create test input file
    input_path = Path(temp_dir) / "no_improve.srt"
    input_path.write_text(sample_srt_content)
    
    output_path = Path(temp_dir) / "improve.srt"
    
    result = improver.improve_text(str(input_path), str(output_path))
    assert result == True
    assert output_path.exists()

def test_improve_text_file_not_found(improver):
    result = improver.improve_text(
        "nonexistent.srt",
        "output.srt"
    )
    assert result == False



