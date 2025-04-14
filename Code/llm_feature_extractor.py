import torch
from transformers import pipeline
import sys
import os

# 添加botsay-main路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../botsay-main')))
import lm_utils

class LLMFeatureExtractor:
    def __init__(self, llm_model_name="mistral", base_feature_extractor=None, device=0):
        """
        初始化LLM特征提取器
        
        参数:
        - llm_model_name: 使用的LLM模型名称，支持"mistral", "llama2_7b", "llama2_13b", "llama2_70b", "chatgpt"
        - base_feature_extractor: 基础特征提取器，如RoBERTa
        - device: 设备ID
        """
        self.llm_model_name = llm_model_name
        
        # 如果未提供基础特征提取器，则初始化一个
        if base_feature_extractor is None:
            self.base_feature_extractor = pipeline('feature-extraction', 
                                                  model='roberta-base', 
                                                  tokenizer='roberta-base', 
                                                  device=device, 
                                                  padding=True, 
                                                  truncation=True, 
                                                  max_length=50, 
                                                  add_special_tokens=True)
        else:
            self.base_feature_extractor = base_feature_extractor
        
        # 初始化LLM模型
        lm_utils.llm_init(llm_model_name)
        
    def extract_llm_enhanced_features(self, user_description, tweet_text=None):
        """
        提取LLM增强的用户文本特征
        
        参数:
        - user_description: 用户描述文本
        - tweet_text: 用户推文文本列表（可选）
        
        返回:
        - enhanced_features: 增强的特征向量
        """
        # 1. 提取原始RoBERTa特征
        if user_description is None or user_description == "":
            base_features = torch.zeros(768)
        else:
            base_features_raw = torch.Tensor(self.base_feature_extractor(user_description))
            # 平均所有token的特征
            base_features = base_features_raw.mean(dim=1).squeeze()
        
        # 2. 使用LLM分析用户描述，生成增强特征
        if user_description and len(user_description) > 0:
            llm_prompt = self._generate_description_prompt(user_description)
            llm_analysis = lm_utils.llm_response(llm_prompt, self.llm_model_name)
            
            # 提取LLM分析结果的特征
            llm_features_raw = torch.Tensor(self.base_feature_extractor(llm_analysis))
            llm_features = llm_features_raw.mean(dim=1).squeeze()
        else:
            llm_features = torch.zeros(768)
        
        # 3. 如果提供了推文，分析推文
        if tweet_text and len(tweet_text) > 0:
            tweets = " ".join(tweet_text[:5])  # 只使用前5条推文
            tweet_prompt = self._generate_tweet_prompt(tweets)
            tweet_analysis = lm_utils.llm_response(tweet_prompt, self.llm_model_name)
            
            # 提取推文分析结果的特征
            tweet_features_raw = torch.Tensor(self.base_feature_extractor(tweet_analysis))
            tweet_features = tweet_features_raw.mean(dim=1).squeeze()
        else:
            tweet_features = torch.zeros(768)
        
        # 4. 融合所有特征
        # 简单拼接特征向量
        enhanced_features = torch.cat([base_features, llm_features, tweet_features], dim=0)
        
        return enhanced_features
    
    def extract_llm_enhanced_batch(self, descriptions, tweets_list=None):
        """
        批量提取增强特征
        
        参数:
        - descriptions: 用户描述列表
        - tweets_list: 用户推文列表的列表
        
        返回:
        - batch_features: 批量增强特征张量
        """
        batch_features = []
        
        for i, desc in enumerate(descriptions):
            # 获取当前用户的推文
            tweets = None
            if tweets_list and i < len(tweets_list):
                tweets = tweets_list[i]
            
            # 提取增强特征
            features = self.extract_llm_enhanced_features(desc, tweets)
            batch_features.append(features)
        
        # 转换为张量并返回
        return torch.stack(batch_features)
    
    def _generate_description_prompt(self, description):
        """生成分析用户描述的提示"""
        prompt = f"""
Analyze this Twitter user description and extract key behavioral patterns that might indicate if this is a human or bot account:

Description: {description}

First, identify any suspicious patterns in the text, such as:
1. Presence of many hashtags, URLs, or mentions
2. Usage of repetitive or generic phrases
3. Lack of personal information
4. Automated or commercial-sounding content
5. Unusual language patterns or errors

Then, summarize what this description suggests about the account being a human or bot. Focus only on text patterns in your analysis.
"""
        return prompt.strip()
    
    def _generate_tweet_prompt(self, tweets):
        """生成分析用户推文的提示"""
        prompt = f"""
Analyze these Twitter posts from a single user and extract patterns that might indicate if this is a human or bot account:

Tweets: {tweets}

Specifically look for:
1. Posting frequency patterns (if evident)
2. Content similarity across tweets
3. Usage of hashtags, URLs, and mentions
4. Engagement with trends or other users
5. Emotional variety and authenticity in tone
6. Evidence of automation or copy-pasting

Summarize what these posts suggest about the account being a human or bot. Focus only on text patterns.
"""
        return prompt.strip()
    
    def close(self):
        """清理资源"""
        lm_utils.wipe_model() 
