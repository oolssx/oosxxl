import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
import numpy as np


class NodeType(Enum):
    CREDIT_FACTOR = "credit_factor"
    USER_ACTION = "user_action"
    CREDIT_SCORE = "credit_score"
    FINANCIAL_PRODUCT = "financial_product"
    INSTITUTION = "institution"
    CONSEQUENCE = "consequence"


class RelationType(Enum):
    AFFECTS = "affects"
    CAUSED_BY = "caused_by"
    REQUIRES = "requires"
    IMPROVES = "improves"
    WORSENS = "worsens"
    CORRELATES_WITH = "correlates_with"
    DEPENDS_ON = "depends_on"
    MITIGATES = "mitigates"


@dataclass
class GraphNode:
    id: str
    type: NodeType
    name: str
    description: str
    weight: float
    metadata: Dict


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: RelationType
    weight: float
    confidence: float
    metadata: Dict


class CreditKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_index = {}
        self.edge_index = defaultdict(list)
        self.impact_cache = {}
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        self._add_credit_factors()
        self._add_user_actions()
        self._add_financial_products()
        self._add_consequences()
        self._add_relationships()
    
    def _add_credit_factors(self):
        factors = [
            {
                'id': 'payment_history',
                'name': '支付历史',
                'description': '按时还款的历史记录',
                'weight': 0.35,
                'metadata': {
                    'category': 'behavioral',
                    'timeframe': 'long_term',
                    'volatility': 'low'
                }
            },
            {
                'id': 'credit_utilization',
                'name': '信用使用率',
                'description': '已用信用额度占总额度的比例',
                'weight': 0.30,
                'metadata': {
                    'category': 'utilization',
                    'timeframe': 'short_term',
                    'volatility': 'high',
                    'optimal_range': [0.1, 0.3]
                }
            },
            {
                'id': 'credit_history_length',
                'name': '信用历史长度',
                'description': '信用账户的平均年龄',
                'weight': 0.15,
                'metadata': {
                    'category': 'history',
                    'timeframe': 'long_term',
                    'volatility': 'very_low'
                }
            },
            {
                'id': 'credit_mix',
                'name': '信用组合',
                'description': '不同类型信用账户的多样性',
                'weight': 0.10,
                'metadata': {
                    'category': 'diversity',
                    'timeframe': 'medium_term',
                    'volatility': 'low'
                }
            },
            {
                'id': 'new_credit',
                'name': '新开账户',
                'description': '最近开设的信用账户数量',
                'weight': 0.10,
                'metadata': {
                    'category': 'inquiry',
                    'timeframe': 'short_term',
                    'volatility': 'medium'
                }
            },
            {
                'id': 'hard_inquiries',
                'name': '硬查询次数',
                'description': '贷款或信用卡申请导致的征信查询',
                'weight': 0.08,
                'metadata': {
                    'category': 'inquiry',
                    'timeframe': 'short_term',
                    'volatility': 'high',
                    'decay_months': 24
                }
            },
            {
                'id': 'debt_to_income',
                'name': '负债收入比',
                'description': '总债务相对于收入的比例',
                'weight': 0.12,
                'metadata': {
                    'category': 'capacity',
                    'timeframe': 'current',
                    'volatility': 'medium'
                }
            },
            {
                'id': 'account_age_diversity',
                'name': '账户年龄多样性',
                'description': '新旧账户的平衡程度',
                'weight': 0.05,
                'metadata': {
                    'category': 'stability',
                    'timeframe': 'long_term',
                    'volatility': 'low'
                }
            }
        ]
        
        for factor in factors:
            self.add_node(
                GraphNode(
                    id=factor['id'],
                    type=NodeType.CREDIT_FACTOR,
                    name=factor['name'],
                    description=factor['description'],
                    weight=factor['weight'],
                    metadata=factor['metadata']
                )
            )
    
    def _add_user_actions(self):
        actions = [
            {
                'id': 'apply_credit_card',
                'name': '申请信用卡',
                'description': '向银行申请新的信用卡',
                'weight': 0.6,
                'metadata': {
                    'impact_duration': 'short_to_medium',
                    'reversibility': 'low',
                    'common_scenarios': ['increase_credit_limit', 'reward_programs']
                }
            },
            {
                'id': 'apply_loan',
                'name': '申请贷款',
                'description': '申请个人贷款或消费贷款',
                'weight': 0.8,
                'metadata': {
                    'impact_duration': 'medium_to_long',
                    'reversibility': 'very_low',
                    'common_scenarios': ['emergency_funds', 'large_purchase']
                }
            },
            {
                'id': 'close_account',
                'name': '关闭账户',
                'description': '关闭现有的信用账户',
                'weight': 0.5,
                'metadata': {
                    'impact_duration': 'long',
                    'reversibility': 'very_low',
                    'common_scenarios': ['annual_fee_avoidance', 'simplification']
                }
            },
            {
                'id': 'miss_payment',
                'name': '逾期还款',
                'description': '未能按时还款',
                'weight': 0.9,
                'metadata': {
                    'impact_duration': 'very_long',
                    'reversibility': 'very_low',
                    'severity_levels': ['30_days', '60_days', '90_days', '120_days']
                }
            },
            {
                'id': 'pay_off_debt',
                'name': '还清债务',
                'description': '提前或按时还清贷款',
                'weight': 0.7,
                'metadata': {
                    'impact_duration': 'medium',
                    'reversibility': 'none',
                    'positive_action': True
                }
            },
            {
                'id': 'increase_utilization',
                'name': '增加信用使用',
                'description': '使用更多的可用信用额度',
                'weight': 0.6,
                'metadata': {
                    'impact_duration': 'short',
                    'reversibility': 'high',
                    'threshold_warnings': [0.5, 0.7, 0.9]
                }
            },
            {
                'id': 'request_limit_increase',
                'name': '申请提额',
                'description': '要求增加信用卡额度',
                'weight': 0.4,
                'metadata': {
                    'impact_duration': 'short',
                    'reversibility': 'medium',
                    'positive_potential': True
                }
            },
            {
                'id': 'consolidate_debt',
                'name': '债务整合',
                'description': '将多笔债务合并为一笔',
                'weight': 0.5,
                'metadata': {
                    'impact_duration': 'medium',
                    'reversibility': 'low',
                    'strategy': 'optimization'
                }
            }
        ]
        
        for action in actions:
            self.add_node(
                GraphNode(
                    id=action['id'],
                    type=NodeType.USER_ACTION,
                    name=action['name'],
                    description=action['description'],
                    weight=action['weight'],
                    metadata=action['metadata']
                )
            )
    
    def _add_financial_products(self):
        products = [
            {
                'id': 'mortgage',
                'name': '住房贷款',
                'weight': 0.9,
                'metadata': {'credit_requirement': 'high', 'amount_range': 'very_high'}
            },
            {
                'id': 'auto_loan',
                'name': '汽车贷款',
                'weight': 0.7,
                'metadata': {'credit_requirement': 'medium', 'amount_range': 'medium'}
            },
            {
                'id': 'personal_loan',
                'name': '个人贷款',
                'weight': 0.6,
                'metadata': {'credit_requirement': 'medium', 'amount_range': 'medium'}
            },
            {
                'id': 'credit_card_premium',
                'name': '高端信用卡',
                'weight': 0.5,
                'metadata': {'credit_requirement': 'high', 'benefits': 'extensive'}
            }
        ]
        
        for product in products:
            self.add_node(
                GraphNode(
                    id=product['id'],
                    type=NodeType.FINANCIAL_PRODUCT,
                    name=product['name'],
                    description='',
                    weight=product['weight'],
                    metadata=product['metadata']
                )
            )
    
    def _add_consequences(self):
        consequences = [
            {
                'id': 'loan_rejection',
                'name': '贷款被拒',
                'weight': 0.8,
                'metadata': {'severity': 'high', 'duration': 'medium'}
            },
            {
                'id': 'high_interest_rate',
                'name': '高利率',
                'weight': 0.7,
                'metadata': {'severity': 'medium', 'duration': 'long'}
            },
            {
                'id': 'low_credit_limit',
                'name': '低信用额度',
                'weight': 0.5,
                'metadata': {'severity': 'medium', 'duration': 'medium'}
            },
            {
                'id': 'premium_access',
                'name': '优质金融产品准入',
                'weight': 0.6,
                'metadata': {'severity': 'positive', 'duration': 'ongoing'}
            }
        ]
        
        for consequence in consequences:
            self.add_node(
                GraphNode(
                    id=consequence['id'],
                    type=NodeType.CONSEQUENCE,
                    name=consequence['name'],
                    description='',
                    weight=consequence['weight'],
                    metadata=consequence['metadata']
                )
            )
    
    def _add_relationships(self):
        relationships = [
            ('apply_credit_card', 'hard_inquiries', RelationType.AFFECTS, 0.8, 0.95),
            ('apply_credit_card', 'new_credit', RelationType.AFFECTS, 0.7, 0.95),
            ('apply_credit_card', 'credit_mix', RelationType.IMPROVES, 0.4, 0.7),
            ('apply_loan', 'hard_inquiries', RelationType.AFFECTS, 0.9, 0.95),
            ('apply_loan', 'debt_to_income', RelationType.WORSENS, 0.8, 0.9),
            ('close_account', 'credit_history_length', RelationType.WORSENS, 0.6, 0.8),
            ('close_account', 'credit_utilization', RelationType.WORSENS, 0.7, 0.85),
            ('miss_payment', 'payment_history', RelationType.WORSENS, 0.95, 0.99),
            ('pay_off_debt', 'debt_to_income', RelationType.IMPROVES, 0.8, 0.9),
            ('pay_off_debt', 'credit_utilization', RelationType.IMPROVES, 0.7, 0.85),
            ('increase_utilization', 'credit_utilization', RelationType.WORSENS, 0.9, 0.95),
            ('request_limit_increase', 'hard_inquiries', RelationType.AFFECTS, 0.3, 0.6),
            ('request_limit_increase', 'credit_utilization', RelationType.IMPROVES, 0.5, 0.7),
            ('payment_history', 'mortgage', RelationType.REQUIRES, 0.9, 0.95),
            ('credit_utilization', 'mortgage', RelationType.REQUIRES, 0.7, 0.85),
            ('debt_to_income', 'mortgage', RelationType.REQUIRES, 0.8, 0.9),
            ('payment_history', 'credit_card_premium', RelationType.REQUIRES, 0.8, 0.9),
            ('hard_inquiries', 'loan_rejection', RelationType.CAUSED_BY, 0.6, 0.75),
            ('payment_history', 'high_interest_rate', RelationType.MITIGATES, 0.7, 0.85),
            ('credit_utilization', 'low_credit_limit', RelationType.CAUSED_BY, 0.6, 0.8),
        ]
        
        for source, target, relation, weight, confidence in relationships:
            self.add_edge(
                GraphEdge(
                    source=source,
                    target=target,
                    relation=relation,
                    weight=weight,
                    confidence=confidence,
                    metadata={}
                )
            )
    
    def add_node(self, node: GraphNode):
        self.graph.add_node(
            node.id,
            type=node.type,
            name=node.name,
            description=node.description,
            weight=node.weight,
            metadata=node.metadata
        )
        self.node_index[node.id] = node
    
    def add_edge(self, edge: GraphEdge):
        self.graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            weight=edge.weight,
            confidence=edge.confidence,
            metadata=edge.metadata
        )
        self.edge_index[edge.source].append(edge)
    
    def calculate_impact_chain(self, action_id: str, depth: int = 3) -> Dict:
        if action_id not in self.node_index:
            return {'error': 'Action not found'}
        
        cache_key = f"{action_id}_{depth}"
        if cache_key in self.impact_cache:
            return self.impact_cache[cache_key]
        
        impact_chain = {
            'action': action_id,
            'direct_impacts': [],
            'indirect_impacts': [],
            'total_impact_score': 0.0
        }
        
        visited = set()
        
        def explore_impacts(node_id: str, current_depth: int, cumulative_weight: float):
            if current_depth > depth or node_id in visited:
                return
            
            visited.add(node_id)
            
            for edge in self.edge_index.get(node_id, []):
                target_node = self.node_index.get(edge.target)
                if not target_node:
                    continue
                
                impact_weight = edge.weight * cumulative_weight * edge.confidence
                
                impact_info = {
                    'factor': edge.target,
                    'factor_name': target_node.name,
                    'relation': edge.relation.value,
                    'weight': impact_weight,
                    'confidence': edge.confidence,
                    'depth': current_depth
                }
                
                if current_depth == 1:
                    impact_chain['direct_impacts'].append(impact_info)
                else:
                    impact_chain['indirect_impacts'].append(impact_info)
                
                impact_chain['total_impact_score'] += impact_weight * target_node.weight
                
                explore_impacts(edge.target, current_depth + 1, impact_weight)
        
        explore_impacts(action_id, 1, 1.0)
        
        impact_chain['direct_impacts'].sort(key=lambda x: x['weight'], reverse=True)
        impact_chain['indirect_impacts'].sort(key=lambda x: x['weight'], reverse=True)
        
        self.impact_cache[cache_key] = impact_chain
        
        return impact_chain
    
    def find_mitigation_strategies(self, negative_factors: List[str]) -> List[Dict]:
        strategies = []
        
        for factor in negative_factors:
            if factor not in self.node_index:
                continue
            
            incoming_edges = [
                (edge, self.node_index[edge.source])
                for source_id in self.node_index.keys()
                for edge in self.edge_index.get(source_id, [])
                if edge.target == factor and edge.relation in [RelationType.IMPROVES, RelationType.MITIGATES]
            ]
            
            for edge, source_node in incoming_edges:
                if source_node.type == NodeType.USER_ACTION:
                    strategies.append({
                        'factor': factor,
                        'action': source_node.id,
                        'action_name': source_node.name,
                        'effectiveness': edge.weight * edge.confidence,
                        'description': source_node.description,
                        'metadata': source_node.metadata
                    })
        
        strategies.sort(key=lambda x: x['effectiveness'], reverse=True)
        
        return strategies
    
    def find_prerequisite_factors(self, product_id: str) -> List[Dict]:
        if product_id not in self.node_index:
            return []
        
        prerequisites = []
        
        for source_id in self.node_index.keys():
            for edge in self.edge_index.get(source_id, []):
                if edge.target == product_id and edge.relation == RelationType.REQUIRES:
                    source_node = self.node_index[source_id]
                    
                    if source_node.type == NodeType.CREDIT_FACTOR:
                        prerequisites.append({
                            'factor_id': source_id,
                            'factor_name': source_node.name,
                            'importance': edge.weight,
                            'confidence': edge.confidence,
                            'description': source_node.description,
                            'optimal_range': source_node.metadata.get('optimal_range')
                        })
        
        prerequisites.sort(key=lambda x: x['importance'], reverse=True)
        
        return prerequisites
    
    def simulate_action_sequence(self, actions: List[Tuple[str, int]]) -> Dict:
        simulation_result = {
            'actions': [],
            'cumulative_impacts': defaultdict(float),
            'timeline': []
        }
        
        current_state = {factor_id: 0.0 for factor_id in self.node_index.keys() 
                        if self.node_index[factor_id].type == NodeType.CREDIT_FACTOR}
        
        for action_id, time_offset in actions:
            impact_chain = self.calculate_impact_chain(action_id, depth=2)
            
            action_result = {
                'action': action_id,
                'time': time_offset,
                'impacts': []
            }
            
            for impact in impact_chain['direct_impacts']:
                factor_id = impact['factor']
                
                if impact['relation'] == 'worsens':
                    change = -impact['weight']
                elif impact['relation'] in ['improves', 'mitigates']:
                    change = impact['weight']
                else:
                    change = impact['weight'] * 0.5
                
                current_state[factor_id] += change
                simulation_result['cumulative_impacts'][factor_id] += change
                
                action_result['impacts'].append({
                    'factor': factor_id,
                    'change': change,
                    'new_value': current_state[factor_id]
                })
            
            simulation_result['actions'].append(action_result)
            
            simulation_result['timeline'].append({
                'time': time_offset,
                'state': current_state.copy(),
                'action': action_id
            })
        
        return simulation_result
    
    def get_factor_relationships(self, factor_id: str) -> Dict:
        if factor_id not in self.node_index:
            return {'error': 'Factor not found'}
        
        relationships = {
            'factor': factor_id,
            'affects': [],
            'affected_by': [],
            'correlates_with': []
        }
        
        for edge in self.edge_index.get(factor_id, []):
            relationships['affects'].append({
                'target': edge.target,
                'target_name': self.node_index[edge.target].name,
                'relation': edge.relation.value,
                'strength': edge.weight
            })
        
        for source_id in self.node_index.keys():
            for edge in self.edge_index.get(source_id, []):
                if edge.target == factor_id:
                    relationships['affected_by'].append({
                        'source': source_id,
                        'source_name': self.node_index[source_id].name,
                        'relation': edge.relation.value,
                        'strength': edge.weight
                    })
        
        return relationships
    
    def find_shortest_improvement_path(self, current_factors: Dict[str, float], 
                                       target_product: str) -> List[Dict]:
        prerequisites = self.find_prerequisite_factors(target_product)
        
        improvement_actions = []
        
        for prereq in prerequisites:
            factor_id = prereq['factor_id']
            current_value = current_factors.get(factor_id, 0.5)
            
            if current_value < 0.7:
                mitigation_strategies = self.find_mitigation_strategies([factor_id])
                
                for strategy in mitigation_strategies[:2]:
                    improvement_actions.append({
                        'factor': factor_id,
                        'factor_name': prereq['factor_name'],
                        'current_value': current_value,
                        'action': strategy['action_name'],
                        'expected_improvement': strategy['effectiveness'],
                        'priority': prereq['importance'] * (1 - current_value)
                    })
        
        improvement_actions.sort(key=lambda x: x['priority'], reverse=True)
        
        return improvement_actions
    
    def export_graph(self, filename: str):
        graph_data = {
            'nodes': [
                {
                    'id': node_id,
                    'type': node.type.value,
                    'name': node.name,
                    'description': node.description,
                    'weight': node.weight,
                    'metadata': node.metadata
                }
                for node_id, node in self.node_index.items()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'relation': edge.relation.value,
                    'weight': edge.weight,
                    'confidence': edge.confidence
                }
                for source_id in self.edge_index.keys()
                for edge in self.edge_index[source_id]
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    def get_graph_statistics(self) -> Dict:
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {
                node_type.value: sum(
                    1 for node in self.node_index.values() 
                    if node.type == node_type
                )
                for node_type in NodeType
            },
            'relation_types': {
                relation_type.value: sum(
                    1 for edges in self.edge_index.values()
                    for edge in edges
                    if edge.relation == relation_type
                )
                for relation_type in RelationType
            },
            'average_node_degree': sum(
                self.graph.degree(node) for node in self.graph.nodes()
            ) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }