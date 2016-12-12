# 对话机器人平台: wit.ai和yige.ai调研

## 概览
wit.ai和yige.ai允许用户预定义一组问题和一些实体来build一个对话机器人.在wit.ai和yige.ai中有三种角色: 平台,机器人的开发者, 机器人的用户. 平台提供一个机器人的框架, 机器人的开发者定义一些少量的数据和少量的代码便可以建立一个个性化的对话机器人. 这个对话机器人服务的对象是机器人的用户.本篇文档通过探索这两者提供的api及输入输出参数, 来回答这些问题: 如果我们要做一个类似的系统,系统框架是怎样的? 框架里的每个模块都有什么功能?


## 传统对话系统框架
> 注:这一小节内容是speech and language processing 2nd edition ch 24 Dialogue and Conversational Agents 的笔记, 如果有细节不清楚的地方请查阅书籍原章节.

一个典型的对话系统架构如下图所示

![conversational agent architecture](/assets/img/conversational-agents.png)

系统主要由6个组件构成. 其中Speech Recognition(ASR)和Natural Language Understanding(NLU)主要是从用户的输入中抽取语义信息, 而Natural Language Generation(NLG)和Text-to-Speech Synthesis(TTS)模块则是讲语义信息转换成回答. Dialogue Manager控制整个对话过程, Task Manager则维护具体领域的相关知识(如在一个天气机器人中维护所有地点的天气信息). 接下来, 我们会大体过一遍这些模块以及其负责的事情, 考虑到之后的场景处理的更多是文本输入和文本输出, 将会跳过ASR和TTS模块.

### NLU
NLU(语义理解)模块将用户的输入转换成对话系统领域相关的语义表示. 这个语义表示可以是Frame-and-slot semantics, 如在一个购票对话系统中,用户的query均可以表示为:

```javascript
ticket:
  flight:
    origin:
        city: Beijing
        date: 2016/11/07 10:30  
    dest:
        city: Shanghai
```
大多数的task-related dialogue system的语义表示都可以是Frame-and-slot semantics, 因为这些系统主要是为了解决类似订票,买咖啡等类似的问题, 系统只需要识别task相关的参数便可以完成整个任务.

生成Frame-and-slot semantics的方法有很多, 如基于semantic grammar, semantic HMM等.

### Response Generation
Response Generation模块选择要表达给用户的概念(concepts), 并且计划用文本表达这些概念的表达方式.
Generation task主要可以分为两个任务:*what to say*, 和*how to say it*. **Content planer**模块
主要针对地一个问题, 决定最终展示给用户的内容, 如是问一个问题,还是直接给出一个答案. Content planning
模块一般都被合并到Dialogue manager中.

Language generation模块则是针对第二个任务,选择合适的句法结构和词语来表达所需要表达的意思. 实现
language generation主要有两种方式: template-based generation和natural language generation. 在template-
based generation中, 答案是一些预定义好的模板. 这些模板的大部分的词语都是固定好的,只有少部分的
变量需要被generator赋值. 例如:
```
What time do you want to leave CITY-ORIG??
Will you return to CITY-ORIG from CITY-DEST?
```
基于natural language generation的方式则接受dialogue manager(content planner)生成的需要表达给
用户的意思的*表示(representation)*,直接生成答案.

### Dialogue Manager
Dialogue Manager控制整个对话的结构. The dialogue manager takes input from the ASR/NLU components, maintains some sort of state, interfaces with the task manager, and passes output to the NLG/TTS modules.
                                                                                                                                                                                                                                          
## yige.ai



