import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json


class GoalType(Enum):
    SCORE_TARGET = "score_target"
    MORTGAGE_APPROVAL = "mortgage_approval"
    PREMIUM_CARD = "premium_card"
    LOAN_APPROVAL = "loan_approval"
    LOWER_INTEREST = "lower_interest"


class ActionPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ImprovementAction:
    action_id: str
    title: str
    description: str
    category: str
    priority: ActionPriority
    estimated_impact: float
    time_to_impact: int
    effort_level: str
    cost: float
    dependencies: List[str] = field(default_factory=list)
    detailed_steps: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)


@dataclass
class ImprovementPlan:
    user_id: str
    goal: GoalType
    current_score: float
    target_score: float
    timeline_months: int
    actions: List[ImprovementAction]
    milestones: List[Dict]
    success_probability: float
    alternative_strategies: List[Dict]
    created_at: datetime


class CreditImprovementPlanner:
    def __init__(self, predictor_model, knowledge_graph):
        self.predictor = predictor_model
        self.knowledge_graph = knowledge_graph
        self.action_library = self._initialize_action_library()
        self.milestone_templates = self._initialize_milestones()
    
    def _initialize_action_library(self) -> Dict[str, ImprovementAction]:
        return {
            'reduce_utilization': ImprovementAction(
                action_id='reduce_utilization',
                title='降低信用卡使用率',
                description='将信用卡使用率降低到30%以下',
                category='utilization',
                priority=ActionPriority.CRITICAL,
                estimated_impact=25.0,
                time_to_impact=1,
                effort_level='medium',
                cost=0.0,
                detailed_steps=[
                    '计算当前所有信用卡的总使用率',
                    '确定需要还款的金额',
                    '在账单日前分批还款',
                    '保持使用率在30%以下',
                    '设置自动提醒避免再次超标'
                ],
                tips=[
                    '账单日前还款最有效，报告时使用率更低',
                    '可以分多次还款，不必一次性还清',
                    '如果资金紧张，优先还高使用率的卡'
                ],
                common_mistakes=[
                    '等到还款日才还，此时账单已生成',
                    '只关注单张卡的使用率，忽略总体使用率',
                    '还款后立即大额消费，使用率又上升'
                ]
            ),
            'setup_autopay': ImprovementAction(
                action_id='setup_autopay',
                title='设置自动还款',
                description='避免遗忘导致的逾期',
                category='payment_history',
                priority=ActionPriority.HIGH,
                estimated_impact=15.0,
                time_to_impact=1,
                effort_level='easy',
                cost=0.0,
                detailed_steps=[
                    '登录网银或手机银行',
                    '找到信用卡自动还款设置',
                    '选择全额还款或最低还款',
                    '设置扣款账户',
                    '确认设置成功并测试'
                ],
                tips=[
                    '建议设置全额自动还款',
                    '确保扣款账户有足够余额',
                    '设置还款提醒作为双重保险'
                ],
                common_mistakes=[
                    '设置后不检查是否扣款成功',
                    '扣款账户余额不足导致失败',
                    '忘记更新绑定的银行卡'
                ]
            ),
            'dispute_errors': ImprovementAction(
                action_id='dispute_errors',
                title='纠正征信报告错误',
                description='对征信报告中的错误信息提出异议',
                category='credit_repair',
                priority=ActionPriority.HIGH,
                estimated_impact=20.0,
                time_to_impact=2,
                effort_level='medium',
                cost=0.0,
                detailed_steps=[
                    '仔细检查征信报告的每一项',
                    '标记出可疑或错误的记录',
                    '收集证明材料',
                    '向征信机构提交异议申请',
                    '跟进处理进度直到解决'
                ],
                tips=[
                    '常见错误：重复记录、他人信息、已还清但未更新',
                    '保留所有沟通记录',
                    '异议处理通常需要30-45天'
                ],
                common_mistakes=[
                    '没有准备充分的证明材料',
                    '对正确的信息提出异议',
                    '提交后不跟进进度'
                ]
            ),
            'increase_limits': ImprovementAction(
                action_id='increase_limits',
                title='申请提高信用额度',
                description='通过提额降低使用率',
                category='utilization',
                priority=ActionPriority.MEDIUM,
                estimated_impact=12.0,
                time_to_impact=1,
                effort_level='easy',
                cost=0.0,
                detailed_steps=[
                    '选择使用时间较长的信用卡',
                    '确保近期无逾期记录',
                    '通过官方渠道申请提额',
                    '提供收入证明（如需要）',
                    '耐心等待审核结果'
                ],
                tips=[
                    '提额申请不会影响信用评分',
                    '通常使用6个月以上才能申请',
                    '提额后不要立即大额消费'
                ],
                common_mistakes=[
                    '频繁申请提额',
                    '提额后使用率反而增加',
                    '在多张卡同时申请提额'
                ]
            ),
            'diversify_credit': ImprovementAction(
                action_id='diversify_credit',
                title='丰富信用组合',
                description='增加不同类型的信用账户',
                category='credit_mix',
                priority=ActionPriority.LOW,
                estimated_impact=8.0,
                time_to_impact=3,
                effort_level='medium',
                cost=0.0,
                detailed_steps=[
                    '评估当前信用账户类型',
                    '识别缺失的账户类型',
                    '谨慎选择合适的新账户',
                    '避免短期内开设过多账户',
                    '保持所有账户的良好记录'
                ],
                tips=[
                    '信用组合占比不高，不必刻意追求',
                    '有信用卡+贷款的组合即可',
                    '不要为了多样性而多样性'
                ],
                common_mistakes=[
                    '短期内开设多个新账户',
                    '开设不需要的账户类型',
                    '无法管理过多账户'
                ]
            ),
            'pay_down_debt': ImprovementAction(
                action_id='pay_down_debt',
                title='偿还高息债务',
                description='优先还清高利率债务',
                category='debt_management',
                priority=ActionPriority.HIGH,
                estimated_impact=18.0,
                time_to_impact=3,
                effort_level='hard',
                cost=0.0,
                detailed_steps=[
                    '列出所有债务及利率',
                    '按利率从高到低排序',
                    '制定还款计划',
                    '优先偿还高息债务',
                    '考虑债务整合方案'
                ],
                tips=[
                    '使用雪球法或雪崩法还债',
                    '可以考虑余额转移降低利息',
                    '还债同时保持最低还款额'
                ],
                common_mistakes=[
                    '平均分配还款金额',
                    '忽视高息小额债务',
                    '还债期间又增加新债务'
                ]
            ),
            'old_account_active': ImprovementAction(
                action_id='old_account_active',
                title='保持旧账户活跃',
                description='定期使用最老的信用卡',
                category='credit_history',
                priority=ActionPriority.MEDIUM,
                estimated_impact=10.0,
                time_to_impact=1,
                effort_level='easy',
                cost=0.0,
                detailed_steps=[
                    '找出持有时间最长的信用卡',
                    '每月至少使用一次',
                    '设置小额定期支付',
                    '按时全额还款',
                    '避免关闭旧账户'
                ],
                tips=[
                    '可以绑定小额订阅自动扣款',
                    '账户年龄对信用有长期影响',
                    '即使不常用也要保持活跃'
                ],
                common_mistakes=[
                    '长期不用导致账户被冻结',
                    '因为年费关闭旧卡',
                    '只用新卡忽略旧卡'
                ]
            ),
            'negotiate_removal': ImprovementAction(
                action_id='negotiate_removal',
                title='协商删除负面记录',
                description='与债权人协商删除已解决的负面记录',
                category='credit_repair',
                priority=ActionPriority.MEDIUM,
                estimated_impact=22.0,
                time_to_impact=2,
                effort_level='hard',
                cost=0.0,
                detailed_steps=[
                    '识别可协商的负面记录',
                    '准备协商理由和材料',
                    '联系债权人提出请求',
                    '提供还款证明',
                    '要求书面确认删除'
                ],
                tips=[
                    '已还清的逾期记录更容易协商',
                    '态度诚恳，说明特殊情况',
                    '有些机构提供"pay for delete"服务'
                ],
                common_mistakes=[
                    '对无法删除的记录浪费时间',
                    '没有书面确认就认为已删除',
                    '协商时情绪化'
                ]
            )
        }
    
    def _initialize_milestones(self) -> Dict[str, List[Dict]]:
        return {
            'score_improvement': [
                {'month': 1, 'target': 'utilization_below_50', 'description': '信用卡使用率降至50%以下'},
                {'month': 2, 'target': 'utilization_below_30', 'description': '信用卡使用率降至30%以下'},
                {'month': 3, 'target': 'zero_late_payment', 'description': '保持3个月零逾期'},
                {'month': 6, 'target': 'score_increase_20', 'description': '信用评分提升20分'},
                {'month': 9, 'target': 'score_increase_35', 'description': '信用评分提升35分'},
                {'month': 12, 'target': 'score_increase_50', 'description': '信用评分提升50分'}
            ],
            'mortgage_ready': [
                {'month': 1, 'target': 'inquiry_pause', 'description': '停止所有非必要的信用查询'},
                {'month': 2, 'target': 'utilization_optimal', 'description': '信用使用率优化至10-30%'},
                {'month': 3, 'target': 'debt_ratio_improvement', 'description': '负债收入比降至35%以下'},
                {'month': 4, 'target': 'payment_perfect', 'description': '建立完美还款记录'},
                {'month': 6, 'target': 'score_threshold', 'description': '达到贷款评分要求'}
            ]
        }
    
    def generate_improvement_plan(self, user_id: str, current_state: Dict, 
                                 goal: GoalType, target_score: float,
                                 timeline_months: int) -> ImprovementPlan:
        
        current_score = current_state.get('current_score', 650)
        
        score_gap = target_score - current_score
        
        identified_issues = self._identify_improvement_areas(current_state)
        
        selected_actions = self._select_optimal_actions(
            identified_issues, score_gap, timeline_months
        )
        
        prioritized_actions = self._prioritize_actions(
            selected_actions, current_state, timeline_months
        )
        
        milestones = self._generate_milestones(
            goal, current_score, target_score, timeline_months
        )
        
        success_probability = self._calculate_success_probability(
            current_state, target_score, timeline_months, prioritized_actions
        )
        
        alternative_strategies = self._generate_alternatives(
            current_state, goal, target_score
        )
        
        plan = ImprovementPlan(
            user_id=user_id,
            goal=goal,
            current_score=current_score,
            target_score=target_score,
            timeline_months=timeline_months,
            actions=prioritized_actions,
            milestones=milestones,
            success_probability=success_probability,
            alternative_strategies=alternative_strategies,
            created_at=datetime.now()
        )
        
        return plan
    
    def _identify_improvement_areas(self, current_state: Dict) -> List[Dict]:
        issues = []
        
        utilization = current_state.get('credit_utilization', 0)
        if utilization > 0.7:
            issues.append({
                'area': 'utilization',
                'severity': 'critical',
                'current_value': utilization,
                'target_value': 0.3,
                'impact_potential': 25
            })
        elif utilization > 0.5:
            issues.append({
                'area': 'utilization',
                'severity': 'high',
                'current_value': utilization,
                'target_value': 0.3,
                'impact_potential': 15
            })
        
        overdue_count = current_state.get('total_overdue_count', 0)
        if overdue_count > 0:
            issues.append({
                'area': 'payment_history',
                'severity': 'critical',
                'current_value': overdue_count,
                'target_value': 0,
                'impact_potential': 30
            })
        
        hard_inquiries = current_state.get('hard_inquiries_6m', 0)
        if hard_inquiries > 4:
            issues.append({
                'area': 'inquiries',
                'severity': 'medium',
                'current_value': hard_inquiries,
                'target_value': 2,
                'impact_potential': 10
            })
        
        dti = current_state.get('debt_to_income_ratio', 0)
        if dti > 0.4:
            issues.append({
                'area': 'debt_ratio',
                'severity': 'high',
                'current_value': dti,
                'target_value': 0.35,
                'impact_potential': 18
            })
        
        credit_history = current_state.get('credit_history_months', 0)
        if credit_history < 24:
            issues.append({
                'area': 'credit_history',
                'severity': 'low',
                'current_value': credit_history,
                'target_value': 36,
                'impact_potential': 8
            })
        
        return sorted(issues, key=lambda x: x['impact_potential'], reverse=True)
    
    def _select_optimal_actions(self, issues: List[Dict], 
                               score_gap: float, timeline: int) -> List[ImprovementAction]:
        selected = []
        
        action_mapping = {
            'utilization': ['reduce_utilization', 'increase_limits'],
            'payment_history': ['setup_autopay', 'negotiate_removal'],
            'inquiries': [],
            'debt_ratio': ['pay_down_debt'],
            'credit_history': ['old_account_active'],
            'credit_repair': ['dispute_errors']
        }
        
        for issue in issues:
            area = issue['area']
            related_actions = action_mapping.get(area, [])
            
            for action_id in related_actions:
                if action_id in self.action_library:
                    action = self.action_library[action_id]
                    
                    if action.time_to_impact <= timeline:
                        selected.append(action)
        
        if score_gap > 30:
            if 'diversify_credit' in self.action_library and timeline >= 6:
                selected.append(self.action_library['diversify_credit'])
        
        return list({action.action_id: action for action in selected}.values())
    
    def _prioritize_actions(self, actions: List[ImprovementAction], 
                           current_state: Dict, timeline: int) -> List[ImprovementAction]:
        
        scored_actions = []
        
        for action in actions:
            priority_score = 0
            
            if action.priority == ActionPriority.CRITICAL:
                priority_score += 100
            elif action.priority == ActionPriority.HIGH:
                priority_score += 75
            elif action.priority == ActionPriority.MEDIUM:
                priority_score += 50
            else:
                priority_score += 25
            
            priority_score += action.estimated_impact
            
            if action.time_to_impact <= timeline / 3:
                priority_score += 20
            
            if action.effort_level == 'easy':
                priority_score += 15
            elif action.effort_level == 'medium':
                priority_score += 5
            
            scored_actions.append((action, priority_score))
        
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        return [action for action, score in scored_actions]
    
    def _generate_milestones(self, goal: GoalType, current_score: float,
                            target_score: float, timeline: int) -> List[Dict]:
        
        if goal == GoalType.MORTGAGE_APPROVAL:
            template = self.milestone_templates.get('mortgage_ready', [])
        else:
            template = self.milestone_templates.get('score_improvement', [])
        
        score_increment = (target_score - current_score) / timeline
        
        milestones = []
        for item in template:
            if item['month'] <= timeline:
                milestone = item.copy()
                milestone['target_score'] = current_score + (score_increment * item['month'])
                milestone['deadline'] = (datetime.now() + timedelta(days=30 * item['month'])).strftime('%Y-%m-%d')
                milestones.append(milestone)
        
        return milestones
    
    def _calculate_success_probability(self, current_state: Dict, 
                                      target_score: float, timeline: int,
                                      actions: List[ImprovementAction]) -> float:
        
        base_probability = 0.5
        
        score_gap = target_score - current_state.get('current_score', 650)
        
        total_potential_impact = sum(action.estimated_impact for action in actions)
        
        if total_potential_impact >= score_gap * 1.5:
            impact_factor = 0.3
        elif total_potential_impact >= score_gap:
            impact_factor = 0.2
        else:
            impact_factor = -0.2
        
        if timeline >= 12:
            time_factor = 0.2
        elif timeline >= 6:
            time_factor = 0.1
        elif timeline >= 3:
            time_factor = 0.0
        else:
            time_factor = -0.3
        
        utilization = current_state.get('credit_utilization', 0.5)
        if utilization < 0.3:
            condition_factor = 0.2
        elif utilization < 0.5:
            condition_factor = 0.1
        else:
            condition_factor = -0.1
        
        probability = base_probability + impact_factor + time_factor + condition_factor
        
        return max(0.1, min(0.95, probability))
    
    def _generate_alternatives(self, current_state: Dict, 
                              goal: GoalType, target_score: float) -> List[Dict]:
        alternatives = []
        
        alternatives.append({
            'strategy': '激进策略',
            'description': '集中在前3个月采取所有高影响行动',
            'pros': ['快速见效', '短期内达成目标'],
            'cons': ['压力较大', '可能影响日常生活'],
            'suitable_for': '时间紧迫，有充足资金'
        })
        
        alternatives.append({
            'strategy': '稳健策略',
            'description': '均匀分布行动，循序渐进',
            'pros': ['压力小', '更可持续', '成功率高'],
            'cons': ['见效较慢'],
            'suitable_for': '时间充裕，追求稳定'
        })
        
        if current_state.get('total_debt', 0) > 50000:
            alternatives.append({
                'strategy': '债务整合策略',
                'description': '先整合债务降低利息，再逐步提升信用',
                'pros': ['降低还款压力', '长期利益大'],
                'cons': ['需要额外申请', '短期可能影响信用'],
                'suitable_for': '高负债用户'
            })
        
        return alternatives
    
    def track_plan_progress(self, user_id: str, plan: ImprovementPlan, 
                           current_state: Dict) -> Dict:
        
        completed_actions = []
        pending_actions = []
        
        for action in plan.actions:
            is_completed = self._check_action_completion(action, current_state)
            if is_completed:
                completed_actions.append(action.action_id)
            else:
                pending_actions.append(action.action_id)
        
        current_score = current_state.get('current_score', plan.current_score)
        score_improvement = current_score - plan.current_score
        expected_improvement = (plan.target_score - plan.current_score) * (len(completed_actions) / len(plan.actions))
        
        progress_percentage = (score_improvement / (plan.target_score - plan.current_score)) * 100 if plan.target_score > plan.current_score else 100
        
        on_track = score_improvement >= expected_improvement * 0.8
        
        return {
            'user_id': user_id,
            'plan_start_date': plan.created_at.strftime('%Y-%m-%d'),
            'current_score': current_score,
            'target_score': plan.target_score,
            'score_improvement': score_improvement,
            'progress_percentage': min(100, max(0, progress_percentage)),
            'completed_actions': len(completed_actions),
            'total_actions': len(plan.actions),
            'on_track': on_track,
            'next_milestone': self._get_next_milestone(plan.milestones, current_score)
        }
    
    def _check_action_completion(self, action: ImprovementAction, 
                                 current_state: Dict) -> bool:
        
        completion_criteria = {
            'reduce_utilization': current_state.get('credit_utilization', 1.0) < 0.3,
            'setup_autopay': current_state.get('autopay_enabled', False),
            'pay_down_debt': current_state.get('debt_to_income_ratio', 1.0) < 0.35,
            'increase_limits': current_state.get('credit_limit_increased', False),
        }
        
        return completion_criteria.get(action.action_id, False)
    
    def _get_next_milestone(self, milestones: List[Dict], 
                           current_score: float) -> Optional[Dict]:
        for milestone in milestones:
            if current_score < milestone.get('target_score', float('inf')):
                return milestone
        return None