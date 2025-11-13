import anthropic
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from enum import Enum


class ExplanationType(Enum):
    CREDIT_SCORE = "credit_score"
    RISK_WARNING = "risk_warning"
    IMPROVEMENT_PLAN = "improvement_plan"
    FACTOR_ANALYSIS = "factor_analysis"
    SIMULATION_RESULT = "simulation_result"
    MILESTONE_UPDATE = "milestone_update"


class ToneStyle(Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ENCOURAGING = "encouraging"
    URGENT = "urgent"
    EDUCATIONAL = "educational"


class AIExplanationGenerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"
        self.max_tokens = 1024
        self.temperature = 0.7
        self.user_profiles = {}
        self.explanation_cache = {}
    
    async def generate_credit_score_explanation(self, user_data: Dict, 
                                               score_breakdown: Dict,
                                               tone: ToneStyle = ToneStyle.FRIENDLY) -> str:
        
        user_profile = self.user_profiles.get(user_data['user_id'], {})
        
        prompt = self._build_score_explanation_prompt(
            user_data, score_breakdown, user_profile, tone
        )
        
        explanation = await self._call_claude_api(prompt)
        
        return explanation
    
    def _build_score_explanation_prompt(self, user_data: Dict, 
                                       score_breakdown: Dict,
                                       user_profile: Dict,
                                       tone: ToneStyle) -> str:
        
        tone_instructions = {
            ToneStyle.PROFESSIONAL: "用专业但易懂的语言解释",
            ToneStyle.FRIENDLY: "用轻松友好的语气，就像朋友聊天一样",
            ToneStyle.ENCOURAGING: "用鼓励和积极的语气，强调好的方面",
            ToneStyle.URGENT: "用紧迫但不吓人的语气，强调需要注意的问题",
            ToneStyle.EDUCATIONAL: "用教学的方式，详细解释概念和原理"
        }
        
        prompt = f"""你是一个专业的信用健康顾问，需要向用户解释他们的信用评分。

用户信息：
- 当前信用评分：{user_data.get('current_score', 'N/A')}
- 信用历史长度：{user_data.get('credit_history_months', 'N/A')}个月
- 信用卡使用率：{user_data.get('credit_utilization', 0) * 100:.1f}%
- 近6个月查询次数：{user_data.get('hard_inquiries_6m', 'N/A')}次
- 逾期记录：{user_data.get('total_overdue_count', 'N/A')}次

评分构成（SHAP值分析）：
{json.dumps(score_breakdown.get('feature_importance', {}), ensure_ascii=False, indent=2)}

任务要求：
1. {tone_instructions.get(tone, '用友好的语气')}
2. 解释当前评分的含义（优秀/良好/一般/较差）
3. 分析影响评分的主要因素（正面和负面各列举2-3个）
4. 用生活化的比喻帮助理解（比如把信用评分比作健康体检报告）
5. 给出1-2个最关键的改进建议
6. 保持在200-300字之内
7. 避免使用过于专业的金融术语，如果必须使用则简单解释
8. 不要简单重复数据，而是要解释数据背后的意义

请生成信用评分解释："""
        
        return prompt
    
    async def generate_warning_explanation(self, warning_data: Dict, 
                                          user_context: Dict,
                                          tone: ToneStyle = ToneStyle.URGENT) -> str:
        
        prompt = f"""你是用户的信用健康管家，需要就一个潜在的信用风险向用户发出预警。

预警信息：
- 操作类型：{warning_data.get('action_type', 'N/A')}
- 风险等级：{warning_data.get('risk_level', 'N/A')}
- 预计影响：信用评分可能降低{abs(warning_data.get('impact_prediction', {}).get('immediate_impact', 0)):.0f}分
- 恢复时间：约{warning_data.get('impact_prediction', {}).get('recovery_time_estimate', 'N/A')}个月

用户背景：
- 当前信用评分：{user_context.get('current_score', 'N/A')}
- 近期计划：{user_context.get('future_plans', '无特殊计划')}
- 最近查询次数：{user_context.get('hard_inquiries_6m', 0)}次

任务要求：
1. 用紧迫但不吓人的语气，让用户意识到问题的严重性
2. 清楚说明本次操作会带来什么影响
3. 如果用户有重要的近期计划（如买房），特别强调影响
4. 提供2-3个具体的替代建议
5. 解释为什么这个操作有风险（用简单的比喻）
6. 保持在250-350字之内
7. 以问题开头吸引注意（比如"等一下，确定要这样做吗？"）
8. 结尾给予明确的行动建议

请生成预警说明："""
        
        explanation = await self._call_claude_api(prompt)
        
        return explanation
    
    async def generate_improvement_plan_narrative(self, plan_data: Dict,
                                                 user_data: Dict,
                                                 tone: ToneStyle = ToneStyle.ENCOURAGING) -> str:
        
        prompt = f"""你是用户的信用提升教练，需要为用户解释一个个性化的信用提升计划。

计划概况：
- 当前评分：{plan_data.get('current_score', 'N/A')}
- 目标评分：{plan_data.get('target_score', 'N/A')}
- 计划周期：{plan_data.get('timeline_months', 'N/A')}个月
- 成功概率：{plan_data.get('success_probability', 0) * 100:.0f}%

关键行动：
{self._format_actions_for_prompt(plan_data.get('actions', []))}

里程碑：
{self._format_milestones_for_prompt(plan_data.get('milestones', []))}

任务要求：
1. 用鼓励和积极的语气，让用户相信目标可以达成
2. 清楚说明这个计划能帮用户实现什么
3. 突出最重要的1-2个行动，解释为什么它们最关键
4. 描绘一个"时间路线图"，让用户看到变化的过程
5. 承认困难，但强调可行性
6. 用真实感和激励性的语言（比如"就像健身一样，坚持3个月就能看到明显效果"）
7. 保持在300-400字之内
8. 结尾给予信心和具体的第一步行动

请生成计划说明："""
        
        explanation = await self._call_claude_api(prompt)
        
        return explanation
    
    async def generate_factor_analysis(self, factor_name: str,
                                      factor_value: float,
                                      impact_score: float,
                                      user_data: Dict) -> str:
        
        prompt = f"""你是信用知识专家，需要向用户解释一个信用影响因素。

因素信息：
- 因素名称：{factor_name}
- 用户当前值：{factor_value}
- 对信用评分的影响：{impact_score:+.1f}分
- 用户整体评分：{user_data.get('current_score', 'N/A')}

任务要求：
1. 用简单的语言解释这个因素是什么
2. 说明这个因素为什么重要（占信用评分的权重）
3. 解释用户当前的数值意味着什么（好/一般/需要改进）
4. 给出行业标准或建议范围作为参考
5. 提供1-2个改进这个因素的具体方法
6. 用类比帮助理解（比如"信用使用率就像你的体重指数BMI"）
7. 保持在150-200字之内
8. 避免说教，保持客观和建设性

请生成因素分析："""
        
        explanation = await self._call_claude_api(prompt)
        
        return explanation
    
    async def generate_simulation_result_explanation(self, simulation: Dict,
                                                    scenario_description: str,
                                                    user_data: Dict) -> str:
        
        prompt = f"""你是信用模拟分析师，需要向用户解释一个"假如"场景的模拟结果。

模拟场景：{scenario_description}

模拟结果：
- 当前评分：{simulation.get('original_score', 'N/A')}
- 模拟后评分：{simulation.get('new_score', 'N/A')}
- 评分变化：{simulation.get('score_change', 0):+.0f}分
- 短期影响（1个月）：{simulation.get('time_projections', {}).get('1_month', 0):+.0f}分
- 中期影响（6个月）：{simulation.get('time_projections', {}).get('6_months', 0):+.0f}分
- 长期影响（12个月）：{simulation.get('time_projections', {}).get('12_months', 0):+.0f}分

任务要求：
1. 清楚说明"如果这样做"会发生什么
2. 区分短期和长期影响
3. 如果是负面影响，解释为什么会这样
4. 如果是正面影响，鼓励用户采取行动
5. 提供时间维度的分析（多久能恢复或见效）
6. 保持在200-250字之内
7. 用通俗的语言，避免"模拟""预测"等术语
8. 给出明确的"要不要这样做"的建议

请生成模拟结果解释："""
        
        explanation = await self._call_claude_api(prompt)
        
        return explanation
    
    async def generate_milestone_celebration(self, milestone: Dict,
                                            progress_data: Dict,
                                            user_data: Dict) -> str:
        
        prompt = f"""你是用户的信用提升教练，用户刚刚达成了一个重要里程碑，需要给予鼓励和指导。

里程碑信息：
- 达成的目标：{milestone.get('description', 'N/A')}
- 计划时间：第{milestone.get('month', 'N/A')}个月
- 实际达成时间：现在
- 目标评分：{milestone.get('target_score', 'N/A')}
- 实际评分：{progress_data.get('current_score', 'N/A')}

整体进度：
- 完成进度：{progress_data.get('progress_percentage', 0):.0f}%
- 已完成行动：{progress_data.get('completed_actions', 0)}个
- 总行动数：{progress_data.get('total_actions', 0)}个

任务要求：
1. 热烈祝贺用户的成就，表达真诚的认可
2. 具体指出用户做对了什么（完成的关键行动）
3. 强调这个里程碑的意义（距离最终目标更近一步）
4. 提醒下一个里程碑是什么，保持动力
5. 给予继续前进的鼓励，但不要施加压力
6. 保持在150-200字之内
7. 用庆祝的语气，可以用emoji表达情绪
8. 让用户感到自豪和有动力继续

请生成里程碑庆祝消息："""
        
        explanation = await self._call_claude_api(prompt)
        
        return explanation
    
    async def _call_claude_api(self, prompt: str, 
                              system_prompt: Optional[str] = None) -> str:
        
        if not system_prompt:
            system_prompt = """你是一个专业且友好的信用健康顾问，专门帮助用户理解和改善他们的信用状况。

你的特点：
1. 专业但不炫技：用简单的语言解释复杂的金融概念
2. 友好且鼓励：即使指出问题，也用建设性的方式
3. 具体且实用：给出的建议都是可操作的
4. 同理心强：理解用户的焦虑和困惑
5. 避免说教：不用"应该""必须"等强制性语言

你的沟通原则：
- 每个解释都包含"是什么""为什么""怎么办"三个部分
- 用类比和比喻帮助理解
- 避免专业术语，如果必须用则简单解释
- 保持简洁，不要冗长
- 给予希望，让用户相信自己能改善"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            explanation = response.content[0].text
            
            return explanation
        
        except Exception as e:
            return f"生成解释时出现错误：{str(e)}"
    
    def _format_actions_for_prompt(self, actions: List[Dict]) -> str:
        if not actions:
            return "暂无具体行动"
        
        formatted = []
        for i, action in enumerate(actions[:5], 1):
            formatted.append(
                f"{i}. {action.get('title', 'N/A')} - "
                f"预期影响{action.get('estimated_impact', 0):.0f}分"
            )
        
        return '\n'.join(formatted)
    
    def _format_milestones_for_prompt(self, milestones: List[Dict]) -> str:
        if not milestones:
            return "暂无里程碑"
        
        formatted = []
        for milestone in milestones[:3]:
            formatted.append(
                f"第{milestone.get('month', 'N/A')}个月："
                f"{milestone.get('description', 'N/A')}"
            )
        
        return '\n'.join(formatted)
    
    async def generate_comparison_analysis(self, user_data: Dict,
                                          peer_group_data: Dict) -> str:
        
        prompt = f"""你是信用分析师，需要帮助用户理解自己在同类人群中的表现。

用户数据：
- 信用评分：{user_data.get('current_score', 'N/A')}
- 信用使用率：{user_data.get('credit_utilization', 0) * 100:.1f}%
- 信用历史：{user_data.get('credit_history_months', 'N/A')}个月

同类人群（相似年龄和收入）：
- 平均信用评分：{peer_group_data.get('avg_score', 'N/A')}
- 平均信用使用率：{peer_group_data.get('avg_utilization', 0) * 100:.1f}%
- 平均信用历史：{peer_group_data.get('avg_history', 'N/A')}个月

任务要求：
1. 客观分析用户在各个维度的表现（高于/接近/低于平均）
2. 指出用户的优势领域，给予肯定
3. 指出需要改进的领域，但用鼓励的方式
4. 避免让用户感到"不如别人"，强调每个人的情况不同
5. 给出1-2个最容易提升排名的建议
6. 保持在200-250字之内
7. 用"你在这个人群中"的方式描述，而非"你比别人差"
8. 结尾传递积极信息

请生成对比分析："""
        
        explanation = await self._call_claude_api(prompt)
        
        return explanation
    
    async def generate_educational_content(self, topic: str,
                                          user_level: str = "beginner") -> str:
        
        level_descriptions = {
            "beginner": "完全不懂金融知识的小白",
            "intermediate": "有一定了解但不深入",
            "advanced": "比较了解但想要专业知识"
        }
        
        prompt = f"""你是金融知识科普专家，需要向用户讲解信用相关的知识。

知识主题：{topic}
用户水平：{level_descriptions.get(user_level, '初学者')}

任务要求：
1. 根据用户水平调整讲解深度
2. 从基础概念开始，逐步深入
3. 用生活中的例子帮助理解
4. 讲解为什么这个知识重要
5. 给出实际应用场景
6. 避免行业黑话，用人人能懂的语言
7. 保持在250-300字之内
8. 采用"是什么-为什么-怎么用"的结构
9. 穿插一两个"小贴士"或"常见误区"

请生成教育内容："""
        
        explanation = await self._call_claude_api(prompt)
        
        return explanation
    
    def set_user_profile(self, user_id: str, profile: Dict):
        self.user_profiles[user_id] = profile
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        return self.user_profiles.get(user_id)
    
    async def generate_personalized_greeting(self, user_data: Dict,
                                            context: str = "daily_check") -> str:
        
        user_name = user_data.get('name', '朋友')
        current_score = user_data.get('current_score', 0)
        recent_change = user_data.get('score_change_7d', 0)
        
        context_prompts = {
            "daily_check": "日常问候",
            "warning_received": "收到预警后的关心",
            "milestone_achieved": "达成里程碑后的祝贺",
            "plan_start": "开始新计划时的鼓励"
        }
        
        prompt = f"""生成一句简短的个性化问候语。

场景：{context_prompts.get(context, '日常问候')}
用户：{user_name}
当前信用评分：{current_score}
最近变化：{recent_change:+.0f}分（过去7天）

要求：
1. 1-2句话，不超过50字
2. 自然亲切，像朋友打招呼
3. 根据场景调整语气
4. 可以提及信用评分的变化（如果有的话）
5. 不要过于正式或公文化

请生成问候语（只输出问候语本身，不要其他内容）："""
        
        greeting = await self._call_claude_api(prompt)
        
        return greeting.strip()
    
    async def generate_batch_explanations(self, requests: List[Dict]) -> List[str]:
        tasks = []
        
        for request in requests:
            explanation_type = request.get('type')
            
            if explanation_type == ExplanationType.CREDIT_SCORE:
                task = self.generate_credit_score_explanation(
                    request.get('user_data', {}),
                    request.get('score_breakdown', {}),
                    request.get('tone', ToneStyle.FRIENDLY)
                )
            elif explanation_type == ExplanationType.RISK_WARNING:
                task = self.generate_warning_explanation(
                    request.get('warning_data', {}),
                    request.get('user_context', {}),
                    request.get('tone', ToneStyle.URGENT)
                )
            elif explanation_type == ExplanationType.IMPROVEMENT_PLAN:
                task = self.generate_improvement_plan_narrative(
                    request.get('plan_data', {}),
                    request.get('user_data', {}),
                    request.get('tone', ToneStyle.ENCOURAGING)
                )
            else:
                task = None
            
            if task:
                tasks.append(task)
        
        results = []
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                results.append(f"生成失败：{str(e)}")
        
        return results