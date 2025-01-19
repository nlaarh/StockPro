import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import time
import traceback
import plotly.express as px
import math
import statistics
import requests
import json

def print_debug(msg):
    """Print debug message"""
    print(f"DEBUG: {msg}")

def get_raw_financial_data(stock):
    """Get raw financial data from Yahoo Finance API"""
    try:
        # Get raw financial data
        modules = [
            'balanceSheetHistory',
            'balanceSheetHistoryQuarterly',
            'cashflowStatementHistory',
            'cashflowStatementHistoryQuarterly',
            'incomeStatementHistory',
            'incomeStatementHistoryQuarterly',
            'financialData'
        ]
        
        raw_data = stock._download_json_data(modules)
        if not raw_data:
            return {}
        
        data = {}
        
        # Get financial data (most reliable)
        if 'financialData' in raw_data:
            financial = raw_data['financialData']
            data.update({
                'grossMargins': financial.get('grossMargins'),
                'operatingMargins': financial.get('operatingMargins'),
                'profitMargins': financial.get('profitMargins'),
                'returnOnEquity': financial.get('returnOnEquity'),
                'returnOnAssets': financial.get('returnOnAssets'),
                'debtToEquity': financial.get('debtToEquity'),
                'currentRatio': financial.get('currentRatio'),
                'quickRatio': financial.get('quickRatio'),
                'totalCash': financial.get('totalCash'),
                'totalDebt': financial.get('totalDebt'),
                'operatingCashflow': financial.get('operatingCashflow'),
                'freeCashflow': financial.get('freeCashflow')
            })
        
        # Get quarterly balance sheet
        if 'balanceSheetHistoryQuarterly' in raw_data:
            statements = raw_data['balanceSheetHistoryQuarterly'].get('balanceSheetStatements', [])
            if statements:
                latest = statements[0]
                # Balance sheet items
                data['totalStockholderEquity'] = latest.get('totalStockholderEquity')
                data['totalCurrentAssets'] = latest.get('totalCurrentAssets')
                data['totalCurrentLiabilities'] = latest.get('totalCurrentLiabilities')
                data['cash'] = latest.get('cash')
                data['inventory'] = latest.get('inventory')
                data['longTermDebt'] = latest.get('longTermDebt')
                data['shortTermDebt'] = latest.get('shortLongTermDebt', 0)
                data['totalAssets'] = latest.get('totalAssets')
                
                # Previous year balance sheet for growth
                if len(statements) > 3:
                    prev_year = statements[3]
                    data['prevTotalStockholderEquity'] = prev_year.get('totalStockholderEquity')
        
        # Get quarterly income statement
        if 'incomeStatementHistoryQuarterly' in raw_data:
            statements = raw_data['incomeStatementHistoryQuarterly'].get('incomeStatementHistory', [])
            if statements:
                latest = statements[0]
                # Income statement items (annualized)
                data['totalRevenue'] = latest.get('totalRevenue', 0) * 4
                data['netIncome'] = latest.get('netIncome', 0) * 4
                data['operatingIncome'] = latest.get('operatingIncome', 0) * 4
                data['grossProfit'] = latest.get('grossProfit', 0) * 4
                data['ebit'] = latest.get('ebit', 0) * 4
                data['interestExpense'] = latest.get('interestExpense', 0) * 4
        
        # Get quarterly cash flow
        if 'cashflowStatementHistoryQuarterly' in raw_data:
            statements = raw_data['cashflowStatementHistoryQuarterly'].get('cashflowStatements', [])
            if statements:
                latest = statements[0]
                # Cash flow items (annualized)
                data['operatingCashflow'] = latest.get('operatingCashflow', 0) * 4
                data['capitalExpenditures'] = latest.get('capitalExpenditures', 0) * 4
                
                if len(statements) > 3:
                    prev_year = statements[3]
                    data['prevOperatingCashflow'] = prev_year.get('operatingCashflow', 0) * 4
                    data['prevCapitalExpenditures'] = prev_year.get('capitalExpenditures', 0) * 4
        
        return data
    except Exception as e:
        print_debug(f"Error getting raw financial data: {str(e)}")
        return {}

def get_balance_sheet_value(balance, row_names, col_idx=0):
    """Try multiple row names to get a value from balance sheet"""
    if isinstance(row_names, str):
        row_names = [row_names]
    
    for name in row_names:
        if name in balance.index:
            value = balance.loc[name].iloc[col_idx]
            if pd.notna(value) and not np.isinf(value):
                return float(value)
    return None

def calculate_financial_metrics(stock):
    """Calculate financial metrics from statements"""
    metrics = {}
    
    try:
        # Get statements
        income = stock.income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow
        
        # Print debug info
        if not balance.empty:
            print_debug("Balance Sheet Index:")
            print_debug(balance.index.tolist())
        if not income.empty:
            print_debug("Income Statement Index:")
            print_debug(income.index.tolist())
        if not cashflow.empty:
            print_debug("Cash Flow Index:")
            print_debug(cashflow.index.tolist())
        
        # Get market data
        market_metrics = get_market_data(stock)
        metrics.update(market_metrics)
        
        if not income.empty and len(income.columns) > 0:
            # Income statement items
            revenue = get_balance_sheet_value(income, ['Total Revenue', 'Revenue'])
            net_income = get_balance_sheet_value(income, ['Net Income', 'Net Income Common Stockholders'])
            operating_income = get_balance_sheet_value(income, ['Operating Income', 'EBIT'])
            gross_profit = get_balance_sheet_value(income, ['Gross Profit'])
            ebit = get_balance_sheet_value(income, ['EBIT', 'Operating Income'])
            interest_expense = abs(get_balance_sheet_value(income, ['Interest Expense', 'Interest Expense And Debt']) or 0)
            
            # Balance sheet items
            if not balance.empty and len(balance.columns) > 0:
                total_assets = get_balance_sheet_value(balance, ['Total Assets', 'Assets'])
                current_assets = get_balance_sheet_value(balance, ['Total Current Assets', 'Current Assets'])
                current_liabilities = get_balance_sheet_value(balance, ['Total Current Liabilities', 'Current Liabilities'])
                total_equity = get_balance_sheet_value(balance, [
                    'Total Stockholder Equity', 
                    'Stockholders Equity',
                    'Total Stockholders Equity',
                    'Stockholders\' Equity',
                    'Total Equity'
                ])
                inventory = get_balance_sheet_value(balance, ['Inventory', 'Total Inventory']) or 0
                long_term_debt = get_balance_sheet_value(balance, [
                    'Long Term Debt',
                    'Total Long Term Debt',
                    'Long Term Debt And Capital Lease Obligation'
                ]) or 0
                short_term_debt = get_balance_sheet_value(balance, [
                    'Short Term Debt',
                    'Current Debt',
                    'Short Long Term Debt',
                    'Current Portion Of Long Term Debt'
                ]) or 0
                cash = get_balance_sheet_value(balance, [
                    'Cash And Cash Equivalents',
                    'Cash And Short Term Investments',
                    'Cash'
                ])
                
                total_debt = long_term_debt + short_term_debt
                print_debug(f"Long Term Debt: {long_term_debt}")
                print_debug(f"Short Term Debt: {short_term_debt}")
                print_debug(f"Total Debt: {total_debt}")
            
            # Cash flow metrics
            if not cashflow.empty and len(cashflow.columns) > 0:
                operating_cashflow = get_balance_sheet_value(cashflow, [
                    'Operating Cash Flow',
                    'Cash Flow From Operating Activities',
                    'Net Cash Provided By Operating Activities',
                    'Net Cash From Operating Activities',
                    'Net Cash Flow From Operations'
                ])
                capex = abs(get_balance_sheet_value(cashflow, [
                    'Capital Expenditure',
                    'Capital Expenditures',
                    'Purchase Of Plant And Equipment',
                    'Property Plant And Equipment Purchase',
                    'Capex',
                    'Capital Expenditures Acquisition Of Property Plant And Equipment'
                ]) or 0)
                
                print_debug(f"Operating Cash Flow: {operating_cashflow}")
                print_debug(f"Capital Expenditure: {capex}")
                
                # Calculate FCF and ratios
                if operating_cashflow is not None and capex is not None:
                    free_cashflow = operating_cashflow - capex
                    print_debug(f"Free Cash Flow: {free_cashflow}")
                    if free_cashflow != 0:
                        if total_debt:
                            metrics['debtToFCF'] = total_debt / abs(free_cashflow)
                            print_debug(f"Debt to FCF Ratio: {metrics['debtToFCF']}")
                        if metrics.get('marketCap'):
                            metrics['fcfYield'] = free_cashflow / metrics['marketCap']
                            print_debug(f"FCF Yield: {metrics['fcfYield']}")
            
            # Calculate profitability ratios
            if revenue and revenue > 0:
                if gross_profit:
                    metrics['grossMargins'] = gross_profit / revenue
                if operating_income:
                    metrics['operatingMargins'] = operating_income / revenue
                if net_income:
                    metrics['profitMargins'] = net_income / revenue
            
            # Calculate return ratios
            if total_equity and total_equity > 0:
                if net_income:
                    metrics['returnOnEquity'] = net_income / total_equity
                if operating_income:
                    metrics['returnOnInvestment'] = operating_income / total_equity
            
            if total_assets and total_assets > 0 and net_income:
                metrics['returnOnAssets'] = net_income / total_assets
            
            # Calculate liquidity ratios
            if current_liabilities and current_liabilities > 0:
                if current_assets:
                    metrics['currentRatio'] = current_assets / current_liabilities
                if current_assets is not None and inventory is not None:
                    metrics['quickRatio'] = (current_assets - inventory) / current_liabilities
                if cash:
                    metrics['cashRatio'] = cash / current_liabilities
            
            # Calculate solvency ratios
            if total_equity and total_equity > 0 and total_debt:
                metrics['debtToEquity'] = total_debt / total_equity
            
            if interest_expense and interest_expense > 0 and ebit:
                metrics['interestCoverage'] = ebit / interest_expense
            
            # Calculate growth metrics
            if len(income.columns) > 1:
                # Revenue growth
                curr_revenue = get_balance_sheet_value(income, ['Total Revenue', 'Revenue'], 0)
                prev_revenue = get_balance_sheet_value(income, ['Total Revenue', 'Revenue'], 1)
                if curr_revenue and prev_revenue and prev_revenue != 0:
                    metrics['revenueGrowth'] = (curr_revenue - prev_revenue) / abs(prev_revenue)
                
                # Earnings growth
                curr_earnings = get_balance_sheet_value(income, ['Net Income', 'Net Income Common Stockholders'], 0)
                prev_earnings = get_balance_sheet_value(income, ['Net Income', 'Net Income Common Stockholders'], 1)
                if curr_earnings and prev_earnings and prev_earnings != 0:
                    metrics['earningsGrowth'] = (curr_earnings - prev_earnings) / abs(prev_earnings)
            
            if len(balance.columns) > 1:
                # Asset growth
                curr_assets = get_balance_sheet_value(balance, ['Total Assets', 'Assets'], 0)
                prev_assets = get_balance_sheet_value(balance, ['Total Assets', 'Assets'], 1)
                if curr_assets and prev_assets and prev_assets != 0:
                    metrics['assetGrowth'] = (curr_assets - prev_assets) / abs(prev_assets)
                
                # Book value growth
                curr_equity = get_balance_sheet_value(balance, [
                    'Total Stockholder Equity', 
                    'Stockholders Equity',
                    'Total Stockholders Equity',
                    'Stockholders\' Equity',
                    'Total Equity'
                ], 0)
                prev_equity = get_balance_sheet_value(balance, [
                    'Total Stockholder Equity', 
                    'Stockholders Equity',
                    'Total Stockholders Equity',
                    'Stockholders\' Equity',
                    'Total Equity'
                ], 1)
                if curr_equity and prev_equity and prev_equity != 0:
                    metrics['bookValueGrowth'] = (curr_equity - prev_equity) / abs(prev_equity)
            
            if len(cashflow.columns) > 1:
                # FCF growth
                curr_op = get_balance_sheet_value(cashflow, [
                    'Operating Cash Flow',
                    'Cash Flow From Operating Activities',
                    'Net Cash Provided By Operating Activities',
                    'Net Cash From Operating Activities'
                ], 0)
                curr_capex = abs(get_balance_sheet_value(cashflow, [
                    'Capital Expenditure',
                    'Capital Expenditures',
                    'Purchase Of Plant And Equipment',
                    'Property Plant And Equipment Purchase',
                    'Capex'
                ], 0) or 0)
                prev_op = get_balance_sheet_value(cashflow, [
                    'Operating Cash Flow',
                    'Cash Flow From Operating Activities',
                    'Net Cash Provided By Operating Activities',
                    'Net Cash From Operating Activities'
                ], 1)
                prev_capex = abs(get_balance_sheet_value(cashflow, [
                    'Capital Expenditure',
                    'Capital Expenditures',
                    'Purchase Of Plant And Equipment',
                    'Property Plant And Equipment Purchase',
                    'Capex'
                ], 1) or 0)
                
                if all(x is not None for x in [curr_op, curr_capex, prev_op, prev_capex]):
                    curr_fcf = curr_op - curr_capex
                    prev_fcf = prev_op - prev_capex
                    if prev_fcf != 0:
                        metrics['freeCashFlowGrowth'] = (curr_fcf - prev_fcf) / abs(prev_fcf)
    
    except Exception as e:
        print_debug(f"Error calculating financial metrics: {str(e)}")
        print_debug(f"Error type: {type(e)}")
        import traceback
        print_debug(f"Traceback: {traceback.format_exc()}")
    
    return metrics

def get_basic_info(stock, ticker):
    """Get basic stock information using multiple methods"""
    info = {}
    
    # Method 1: Try history (most reliable)
    try:
        hist = stock.history(period="1d")
        if not hist.empty:
            info['currentPrice'] = float(hist['Close'].iloc[-1])
            info['volume'] = float(hist['Volume'].iloc[-1])
            print_debug("Successfully got price from history")
    except Exception as e:
        print_debug(f"Error getting history: {str(e)}")
    
    # Method 2: Try fast_info
    if not info.get('currentPrice'):
        try:
            fast_info = stock.fast_info
            info['currentPrice'] = fast_info.last_price
            info['marketCap'] = fast_info.market_cap
            info['volume'] = fast_info.last_volume
            print_debug("Successfully got data from fast_info")
        except Exception as e:
            print_debug(f"Error getting fast_info: {str(e)}")
    
    # Method 3: Try regular info
    if not info.get('currentPrice'):
        try:
            stock_info = stock.info
            if stock_info:
                info['currentPrice'] = stock_info.get('regularMarketPrice')
                info['volume'] = stock_info.get('regularMarketVolume')
                info['marketCap'] = stock_info.get('marketCap')
                print_debug("Successfully got data from info")
        except Exception as e:
            print_debug(f"Error getting info: {str(e)}")
    
    return info

def get_financial_statements(stock):
    """Get financial statements with fallback methods"""
    try:
        # Try annual statements first
        bs = stock.balance_sheet
        if bs.empty:
            print_debug("Annual balance sheet empty, trying quarterly")
            bs = stock.quarterly_balance_sheet
            
        is_stmt = stock.financials
        if is_stmt.empty:
            print_debug("Annual income statement empty, trying quarterly")
            is_stmt = stock.quarterly_financials
            
        cf = stock.cashflow
        if cf.empty:
            print_debug("Annual cash flow empty, trying quarterly")
            cf = stock.quarterly_cashflow
            
        return bs, is_stmt, cf
    except Exception as e:
        print_debug(f"Error getting financial statements: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def safe_get_value(df, row_name, col_idx=0):
    """Safely get value from DataFrame with error handling"""
    try:
        if row_name in df.index:
            value = df.loc[row_name, df.columns[col_idx]]
            if pd.isna(value) or np.isinf(value):
                return None
            
            # Try alternative row names
            alternatives = {
                'Total Stockholder Equity': ['Total Stockholders\' Equity', 'Stockholders\' Equity', 'Total Equity'],
                'Total Current Assets': ['Current Assets'],
                'Total Current Liabilities': ['Current Liabilities'],
                'Total Assets': ['Assets'],
                'Total Debt': ['Long Term Debt', 'Total Long Term Debt'],
                'Cash': ['Cash And Cash Equivalents', 'Cash & Equivalents'],
                'Inventory': ['Total Inventory', 'Net Inventory']
            }
            
            if row_name in alternatives:
                for alt_name in alternatives[row_name]:
                    if alt_name in df.index:
                        value = df.loc[alt_name, df.columns[col_idx]]
                        if pd.isna(value) or np.isinf(value):
                            continue
                        return float(value)
    except Exception as e:
        print_debug(f"Error getting {row_name}: {str(e)}")
    return None

def get_stock_info(ticker):
    """Get stock information from yfinance with retry mechanism"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = {}
            
            # First get basic info
            basic_info = get_basic_info(stock, ticker)
            info.update(basic_info)
            
            # Calculate financial metrics from statements
            financial_metrics = calculate_financial_metrics(stock)
            info.update(financial_metrics)
            
            # Then try Yahoo Finance API for remaining data
            try:
                url = f"https://query2.finance.yahoo.com/v11/finance/quoteSummary/{ticker}"
                params = {
                    'modules': ','.join([
                        'assetProfile',
                        'defaultKeyStatistics',
                        'summaryDetail',
                        'price'
                    ])
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)'
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=5)
                response.raise_for_status()
                
                data = response.json()
                if 'quoteSummary' in data and 'result' in data['quoteSummary'] and data['quoteSummary']['result']:
                    result = data['quoteSummary']['result'][0]
                    
                    # Get company profile
                    if 'assetProfile' in result:
                        profile = result['assetProfile']
                        profile_fields = ['sector', 'industry', 'country', 'fullTimeEmployees', 'longBusinessSummary']
                        for field in profile_fields:
                            if field in profile:
                                info[field] = profile[field]
                    
                    # Get key statistics
                    if 'defaultKeyStatistics' in result:
                        stats = result['defaultKeyStatistics']
                        metrics = {
                            'forwardPE': 'forwardPE',
                            'priceToBook': 'priceToBook',
                            'pegRatio': 'pegRatio'
                        }
                        for key, stat in metrics.items():
                            if stat in stats and isinstance(stats[stat], dict):
                                raw_value = stats[stat].get('raw')
                                if raw_value is not None:
                                    info[key] = raw_value
                    
                    # Get summary detail
                    if 'summaryDetail' in result:
                        detail = result['summaryDetail']
                        metrics = {
                            'dividendYield': 'dividendYield',
                            'payoutRatio': 'payoutRatio',
                            'priceToSalesTrailing12Months': 'priceToSalesTrailing12Months'
                        }
                        for key, stat in metrics.items():
                            if stat in detail and isinstance(detail[stat], dict):
                                raw_value = detail[stat].get('raw')
                                if raw_value is not None:
                                    info[key] = raw_value
                
                print_debug("Successfully got data from v11 API")
            except requests.exceptions.RequestException as e:
                print_debug(f"Error with API request: {str(e)}")
            except Exception as e:
                print_debug(f"Error processing API data: {str(e)}")
            
            # Get growth metrics
            try:
                # Income Statement Growth
                income = stock.financials
                if not income.empty:
                    current_period = income.columns[0]
                    prev_period = income.columns[1] if len(income.columns) > 1 else None
                    
                    metrics = {
                        'Revenue': 'Total Revenue',
                        'Net Income': 'Net Income',
                        'EBITDA': 'EBITDA'
                    }
                    
                    for metric_name, row_name in metrics.items():
                        if row_name in income.index:
                            try:
                                current_value = float(income.loc[row_name, current_period])
                                if prev_period and not pd.isna(current_value):
                                    prev_value = float(income.loc[row_name, prev_period])
                                    if prev_value and prev_value != 0:
                                        info[f'{metric_name.lower().replace(" ", "")}Growth'] = (current_value - prev_value) / abs(prev_value)
                            except (ValueError, TypeError) as e:
                                print_debug(f"Error processing {row_name}: {str(e)}")
                
                # Balance Sheet Growth
                balance = stock.balance_sheet
                if not balance.empty:
                    current_period = balance.columns[0]
                    prev_period = balance.columns[1] if len(balance.columns) > 1 else None
                    
                    metrics = {
                        'Assets': 'Total Assets',
                        'Equity': 'Total Stockholder Equity'
                    }
                    
                    for metric_name, row_name in metrics.items():
                        if row_name in balance.index:
                            try:
                                current_value = float(balance.loc[row_name, current_period])
                                if prev_period and not pd.isna(current_value):
                                    prev_value = float(balance.loc[row_name, prev_period])
                                    if prev_value and prev_value != 0:
                                        info[f'{metric_name.lower()}Growth'] = (current_value - prev_value) / abs(prev_value)
                            except (ValueError, TypeError) as e:
                                print_debug(f"Error processing {row_name}: {str(e)}")
                
                # Cash Flow Growth
                cashflow = stock.cashflow
                if not cashflow.empty:
                    current_period = cashflow.columns[0]
                    prev_period = cashflow.columns[1] if len(cashflow.columns) > 1 else None
                    
                    if 'Free Cash Flow' in cashflow.index:
                        try:
                            current_fcf = float(cashflow.loc['Free Cash Flow', current_period])
                            if prev_period:
                                prev_fcf = float(cashflow.loc['Free Cash Flow', prev_period])
                                if prev_fcf and prev_fcf != 0:
                                    info['freeCashFlowGrowth'] = (current_fcf - prev_fcf) / abs(prev_fcf)
                        except (ValueError, TypeError) as e:
                            print_debug(f"Error processing Free Cash Flow: {str(e)}")
                
                print_debug("Successfully processed financial statements")
            except Exception as e:
                print_debug(f"Error processing financial statements: {str(e)}")
            
            # Clean up NaN and inf values
            for key in info:
                if isinstance(info[key], (int, float)):
                    if pd.isna(info[key]) or np.isinf(info[key]):
                        info[key] = None
            
            # Verify we have minimum required data
            if not info.get('currentPrice'):
                raise ValueError("Could not retrieve current price")
            
            return info, None
            
        except Exception as e:
            if attempt < max_retries - 1:
                print_debug(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                error_msg = f"Failed to retrieve data for {ticker} after {max_retries} attempts. Error: {str(e)}"
                return None, error_msg
    
    return None, "Maximum retries exceeded"

def get_stock_data(ticker, period="2y"):
    """Safely get stock data with error handling"""
    try:
        stock = yf.download(ticker, period=period)
        return stock, None
    except Exception as e:
        return pd.DataFrame(), str(e)

def calculate_buffett_metrics(info):
    """Calculate Warren Buffett's key metrics and ideal entry price"""
    metrics = {}
    
    try:
        # 1. Return on Equity (should be > 15%)
        if 'returnOnEquity' in info:
            metrics['ROE Analysis'] = {
                'value': info['returnOnEquity'] * 100,
                'threshold': 15,
                'status': '✅' if info['returnOnEquity'] * 100 >= 15 else '❌'
            }
        
        # 2. Profit Margins (should be stable and > 10%)
        if 'profitMargins' in info:
            metrics['Profit Margin Analysis'] = {
                'value': info['profitMargins'] * 100,
                'threshold': 10,
                'status': '✅' if info['profitMargins'] * 100 >= 10 else '❌'
            }
        
        # 3. Low Debt (Debt to Equity < 0.5)
        if 'debtToEquity' in info:
            metrics['Debt Analysis'] = {
                'value': info['debtToEquity'],
                'threshold': 0.5,
                'status': '✅' if info['debtToEquity'] <= 0.5 else '❌'
            }
        
        # 4. Strong Current Ratio (> 1.5)
        if 'currentRatio' in info:
            metrics['Liquidity Analysis'] = {
                'value': info['currentRatio'],
                'threshold': 1.5,
                'status': '✅' if info['currentRatio'] >= 1.5 else '❌'
            }
        
        # 5. Strong Free Cash Flow
        if 'freeCashFlow' in info and 'marketCap' in info:
            fcf_yield = info.get('freeCashFlowYield', 0) * 100
            metrics['Free Cash Flow Analysis'] = {
                'value': fcf_yield,
                'threshold': 6,  # FCF yield should be > 6%
                'status': '✅' if fcf_yield >= 6 else '❌'
            }
        
        # Calculate ideal entry price using multiple methods
        if 'earningsPerShare' in info and 'bookValuePerShare' in info:
            # Method 1: Graham's Number (√(22.5 × EPS × BVPS))
            graham_number = math.sqrt(22.5 * info['earningsPerShare'] * info['bookValuePerShare'])
            
            # Method 2: Conservative DCF with 10% growth rate and 12% discount rate
            if 'freeCashFlow' in info and 'shares' in info:
                fcf_per_share = info['freeCashFlow'] / info['shares']
                growth_rate = 0.10  # 10% growth
                discount_rate = 0.12  # 12% discount rate
                terminal_multiple = 15  # Terminal P/E multiple
                
                future_fcf = fcf_per_share * (1 + growth_rate) ** 10  # 10 years
                terminal_value = future_fcf * terminal_multiple
                dcf_value = terminal_value / (1 + discount_rate) ** 10
                
                # Add present value of FCF for next 10 years
                for i in range(1, 11):
                    yearly_fcf = fcf_per_share * (1 + growth_rate) ** i
                    dcf_value += yearly_fcf / (1 + discount_rate) ** i
            
            # Method 3: Average historical P/E ratio
            conservative_pe = 15  # Conservative P/E ratio
            pe_based_price = info['earningsPerShare'] * conservative_pe
            
            # Take the average of all methods
            ideal_price = statistics.mean([graham_number, dcf_value, pe_based_price])
            current_price = info['currentPrice']
            
            metrics['Ideal Entry Price Analysis'] = {
                'Graham Number': f"${graham_number:.2f}",
                'DCF Value': f"${dcf_value:.2f}",
                'P/E Based': f"${pe_based_price:.2f}",
                'Average Ideal Price': f"${ideal_price:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Margin of Safety': f"{((ideal_price - current_price) / ideal_price * 100):.1f}%"
            }
    
    except Exception as e:
        print_debug(f"Error calculating Buffett metrics: {str(e)}")
    
    return metrics

def display_buffett_analysis(metrics):
    """Display Warren Buffett's analysis in a clean format"""
    st.header("Warren Buffett's Investment Criteria")
    
    # Display key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Metrics Analysis")
        for name, data in metrics.items():
            if name != 'Ideal Entry Price Analysis':
                st.write(f"{name}:")
                st.write(f"Value: {data['value']:.1f}")
                st.write(f"Threshold: {data['threshold']}")
                st.write(f"Status: {data['status']}")
                st.write("---")
    
    with col2:
        if 'Ideal Entry Price Analysis' in metrics:
            st.subheader("Valuation Analysis")
            price_data = metrics['Ideal Entry Price Analysis']
            for key, value in price_data.items():
                st.write(f"{key}: {value}")

def get_stock_news(ticker):
    """Get recent news for the stock"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news
    except Exception as e:
        print_debug(f"Error fetching news: {str(e)}")
        return []

def display_news(news):
    """Display news in a clean format"""
    st.header("Recent News")
    
    for article in news:
        with st.expander(article['title']):
            st.write(f"**Source:** {article.get('publisher', 'Unknown')}")
            st.write(f"**Time:** {datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Summary:** {article.get('summary', 'No summary available')}")
            if article.get('link'):
                st.write(f"[Read More]({article['link']})")

def get_recommendation(info):
    """Generate buy/sell/hold recommendation based on multiple factors"""
    score = 0
    reasons = []
    
    # Valuation Metrics
    if info.get('forwardPE'):
        if info['forwardPE'] < 15:
            score += 2
            reasons.append("Attractive P/E ratio below 15")
        elif info['forwardPE'] > 30:
            score -= 2
            reasons.append("High P/E ratio above 30")
    
    if info.get('priceToBook'):
        if info['priceToBook'] < 3:
            score += 1
            reasons.append("Reasonable Price/Book ratio")
        elif info['priceToBook'] > 5:
            score -= 1
            reasons.append("High Price/Book ratio")
    
    # Growth Metrics
    if info.get('revenueGrowth'):
        if info['revenueGrowth'] > 0.15:
            score += 2
            reasons.append("Strong revenue growth above 15%")
        elif info['revenueGrowth'] < 0:
            score -= 2
            reasons.append("Declining revenue")
    
    if info.get('earningsGrowth'):
        if info['earningsGrowth'] > 0.15:
            score += 2
            reasons.append("Strong earnings growth above 15%")
        elif info['earningsGrowth'] < 0:
            score -= 2
            reasons.append("Declining earnings")
    
    # Financial Health
    if info.get('currentRatio'):
        if info['currentRatio'] > 1.5:
            score += 1
            reasons.append("Strong liquidity position")
        elif info['currentRatio'] < 1:
            score -= 1
            reasons.append("Weak liquidity position")
    
    if info.get('debtToEquity'):
        if info['debtToEquity'] < 0.5:
            score += 1
            reasons.append("Low debt levels")
        elif info['debtToEquity'] > 1:
            score -= 1
            reasons.append("High debt levels")
    
    # Profitability
    if info.get('profitMargins'):
        if info['profitMargins'] > 0.2:
            score += 2
            reasons.append("Strong profit margins above 20%")
        elif info['profitMargins'] < 0.05:
            score -= 2
            reasons.append("Low profit margins")
    
    if info.get('returnOnEquity'):
        if info['returnOnEquity'] > 0.15:
            score += 2
            reasons.append("Strong ROE above 15%")
        elif info['returnOnEquity'] < 0.08:
            score -= 2
            reasons.append("Low ROE")
    
    if info.get('freeCashFlowYield'):
        if info['freeCashFlowYield'] > 0.06:
            score += 2
            reasons.append("Attractive FCF yield above 6%")
        elif info['freeCashFlowYield'] < 0.02:
            score -= 2
            reasons.append("Low FCF yield")
    
    # Determine recommendation
    if score >= 4:
        recommendation = "Strong Buy"
        color = "success"
    elif score >= 2:
        recommendation = "Buy"
        color = "success"
    elif score >= -1:
        recommendation = "Hold"
        color = "warning"
    elif score >= -3:
        recommendation = "Sell"
        color = "danger"
    else:
        recommendation = "Strong Sell"
        color = "danger"
    
    # Get top 3 most important reasons
    if score > 0:
        reasons = [r for r in reasons if "Strong" in r or "Attractive" in r or "Low debt" in r][:3]
    else:
        reasons = [r for r in reasons if "High" in r or "Low" in r or "Declining" in r or "Weak" in r][:3]
    
    return recommendation, color, reasons

def format_market_cap(market_cap):
    """Format market cap to human readable string"""
    if market_cap is None:
        return "N/A"
    elif market_cap >= 1e12:
        return f"${market_cap/1e12:.1f}T"
    elif market_cap >= 1e9:
        return f"${market_cap/1e9:.1f}B"
    elif market_cap >= 1e6:
        return f"${market_cap/1e6:.1f}M"
    else:
        return f"${market_cap:,.0f}"

def format_metric(value):
    """Format numeric metric"""
    if value is None:
        return "N/A"
    return f"{value:.2f}"

def format_percentage(value):
    """Format percentage value"""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"

def format_volume(value):
    """Format volume value"""
    if value is None:
        return "N/A"
    return f"{value:,.0f}"

def display_fundamentals_tab(info):
    """Display fundamentals tab with professional layout"""
    # Company Profile Section
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='margin-bottom: 15px;'>📊 Company Profile</h3>
        </div>
    """, unsafe_allow_html=True)
    
    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        sector = info.get('sector', info.get('industry', 'N/A'))
        industry = info.get('industry', info.get('sector', 'its'))
        st.metric("Sector", sector)
        st.metric("Industry", industry)
    with profile_col2:
        employees = info.get('fullTimeEmployees', info.get('employees', 0))
        country = info.get('country', info.get('headquarters', 'N/A'))
        st.metric("Full Time Employees", f"{employees:,}" if employees else "N/A")
        st.metric("Country", country)

    # Financial Metrics Section
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='margin-bottom: 15px;'>💰 Financial Metrics</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Valuation Metrics
    st.subheader("Valuation")
    val_col1, val_col2, val_col3 = st.columns(3)
    with val_col1:
        pe = info.get('forwardPE')
        st.metric("Forward P/E", format_metric(pe) if pe and pe > 0 else "N/A")
        pb = info.get('priceToBook')
        st.metric("Price/Book", format_metric(pb) if pb and pb > 0 else "N/A")
    with val_col2:
        ps = info.get('priceToSales')
        st.metric("Price/Sales", format_metric(ps) if ps and ps > 0 else "N/A")
        peg = info.get('pegRatio')
        st.metric("PEG Ratio", format_metric(peg) if peg and peg > 0 else "N/A")
    with val_col3:
        ev = info.get('enterpriseToEbitda')
        st.metric("EV/EBITDA", format_metric(ev) if ev and ev > 0 else "N/A")
        fcf_yield = info.get('freeCashFlowYield')
        st.metric("FCF Yield", format_percentage(fcf_yield) if fcf_yield else "N/A")
        market_cap = info.get('marketCap')
        st.metric("Market Cap", format_market_cap(market_cap))

    # Profitability Metrics
    st.subheader("Profitability")
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    with prof_col1:
        gm = info.get('grossMargins')
        st.metric("Gross Margin", format_percentage(gm) if gm else "N/A")
        pm = info.get('profitMargins')
        st.metric("Net Margin", format_percentage(pm) if pm else "N/A")
    with prof_col2:
        om = info.get('operatingMargins')
        st.metric("Operating Margin", format_percentage(om) if om else "N/A")
        roe = info.get('returnOnEquity')
        st.metric("Return on Equity", format_percentage(roe) if roe else "N/A")
    with prof_col3:
        roa = info.get('returnOnAssets')
        st.metric("Return on Assets", format_percentage(roa) if roa else "N/A")
        roic = info.get('returnOnInvestment')
        st.metric("Return on Investment", format_percentage(roic) if roic else "N/A")

    # Growth Metrics
    st.subheader("Growth")
    growth_col1, growth_col2, growth_col3 = st.columns(3)
    with growth_col1:
        rg = info.get('revenueGrowth')
        st.metric("Revenue Growth", format_percentage(rg) if rg else "N/A")
        eg = info.get('earningsGrowth')
        st.metric("Earnings Growth", format_percentage(eg) if eg else "N/A")
    with growth_col2:
        fcfg = info.get('freeCashFlowGrowth')
        st.metric("FCF Growth", format_percentage(fcfg) if fcfg else "N/A")
        dg = info.get('dividendGrowth')
        st.metric("Dividend Growth", format_percentage(dg) if dg else "N/A")
    with growth_col3:
        ag = info.get('assetGrowth')
        st.metric("Asset Growth", format_percentage(ag) if ag else "N/A")
        bg = info.get('bookValueGrowth')
        st.metric("Book Value Growth", format_percentage(bg) if bg else "N/A")

    # Balance Sheet Health
    st.subheader("Balance Sheet Health")
    health_col1, health_col2, health_col3 = st.columns(3)
    with health_col1:
        cr = info.get('currentRatio')
        st.metric("Current Ratio", format_metric(cr) if cr else "N/A")
        qr = info.get('quickRatio')
        st.metric("Quick Ratio", format_metric(qr) if qr else "N/A")
    with health_col2:
        de = info.get('debtToEquity')
        st.metric("Debt/Equity", format_metric(de) if de else "N/A")
        ic = info.get('interestCoverage')
        st.metric("Interest Coverage", format_metric(ic) if ic else "N/A")
    with health_col3:
        cash_ratio = info.get('cashRatio')
        st.metric("Cash Ratio", format_metric(cash_ratio) if cash_ratio else "N/A")
        debt_fcf = info.get('totalDebtToFCF')
        st.metric("Debt/FCF", format_metric(debt_fcf) if debt_fcf else "N/A")

def get_company_summary(info):
    """Generate a concise company summary"""
    try:
        # Get company info
        sector = info.get('sector', 'various sectors')
        industry = info.get('industry', 'various industries')
        employees = info.get('fullTimeEmployees')
        country = info.get('country', 'global markets')
        
        # Format employee count
        if employees:
            try:
                employees = int(employees.replace(',', ''))  # Remove any existing commas
                employees_str = f"{employees:,}"  # Add commas for formatting
            except (ValueError, AttributeError):
                employees_str = "an undisclosed number of"
        else:
            employees_str = "an undisclosed number of"
        
        # Generate summary
        summary = f"""
        A company operating in the {sector} sector, specifically in {industry}. 
        Based in {country}, the company employs {employees_str} people.
        """
        
        return summary.strip()
        
    except Exception as e:
        print_debug(f"Error generating company summary: {str(e)}")
        return "Company information currently unavailable."

def plot_stock_price(data, ticker):
    """Create a candlestick chart"""
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'])])
        
        fig.update_layout(title=f'{ticker} Stock Price',
                         yaxis_title='Price (USD)',
                         template='plotly_white')
        return fig
    except Exception as e:
        print_debug(f"Error creating stock price plot: {str(e)}")
        # Return an empty figure with an error message
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating plot: {str(e)}",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

def predict_stock_price(data, days=30, model_type='linear'):
    """Predict stock prices using selected model"""
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        
        # Prepare data
        df = data.copy()
        df['Prediction'] = df['Close'].shift(-1)
        df = df[['Close', 'Prediction']].dropna()
        
        # Create features and target
        X = np.array(df['Close']).reshape(-1, 1)
        y = np.array(df['Prediction']).reshape(-1, 1)
        
        # Scale the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Select and train model
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100)
        else:  # SVR
            model = SVR(kernel='rbf')
        
        model.fit(X, y.ravel())
        
        # Make prediction
        last_price = data['Close'].iloc[-1]
        prediction = model.predict([[last_price]])[0]
        
        # Create prediction chart
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'],
                               mode='lines',
                               name='Historical Price'))
        
        # Add prediction point
        future_date = data.index[-1] + pd.Timedelta(days=days)
        fig.add_trace(go.Scatter(x=[data.index[-1], future_date],
                               y=[last_price, prediction],
                               mode='lines+markers',
                               line=dict(dash='dash'),
                               name='Prediction'))
        
        fig.update_layout(
            title='Price Prediction',
            yaxis_title='Price (USD)',
            template='plotly_white'
        )
        
        return prediction, fig
        
    except Exception as e:
        print_debug(f"Error in price prediction: {str(e)}")
        return None, None

def get_market_data(stock):
    """Get market data and calculate valuation metrics"""
    metrics = {}
    try:
        # Get market data from fast_info
        fast_info = stock.fast_info
        current_price = None
        
        if hasattr(fast_info, 'market_cap'):
            metrics['marketCap'] = fast_info.market_cap
        
        if hasattr(fast_info, 'last_price'):
            current_price = fast_info.last_price
            metrics['currentPrice'] = current_price
        
        # Get financial statements
        income_stmt = stock.income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow
        
        # Forward P/E using latest EPS
        if not income_stmt.empty and current_price:
            if 'Diluted EPS' in income_stmt.index:
                latest_eps = income_stmt.loc['Diluted EPS'].iloc[0]
                if latest_eps and latest_eps > 0:
                    metrics['forwardPE'] = current_price / latest_eps
        
        # Price/Book
        if not balance.empty and current_price and hasattr(fast_info, 'shares'):
            if 'Stockholders Equity' in balance.index:
                equity = balance.loc['Stockholders Equity'].iloc[0]
                if equity and equity > 0:
                    book_value_per_share = equity / fast_info.shares
                    metrics['priceToBook'] = current_price / book_value_per_share
        
        # Price/Sales
        if not income_stmt.empty and metrics.get('marketCap'):
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                if revenue and revenue > 0:
                    metrics['priceToSales'] = metrics['marketCap'] / revenue
        
        # PEG Ratio using YoY earnings growth
        if not income_stmt.empty and metrics.get('forwardPE'):
            if len(income_stmt.columns) >= 2:  # Need at least 2 periods
                current_eps = income_stmt.loc['Diluted EPS'].iloc[0]
                prev_eps = income_stmt.loc['Diluted EPS'].iloc[1]
                if current_eps and prev_eps and prev_eps != 0:
                    growth_rate = ((current_eps - prev_eps) / abs(prev_eps)) * 100
                    if growth_rate > 0:
                        metrics['pegRatio'] = metrics['forwardPE'] / growth_rate
        
        # EV/EBITDA
        if not income_stmt.empty and not balance.empty and metrics.get('marketCap'):
            if 'EBITDA' in income_stmt.index and 'Total Debt' in balance.index:
                ebitda = income_stmt.loc['EBITDA'].iloc[0]
                total_debt = balance.loc['Total Debt'].iloc[0]
                cash = balance.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance.index else 0
                
                if ebitda and ebitda > 0:
                    enterprise_value = metrics['marketCap'] + total_debt - cash
                    metrics['evToEBITDA'] = enterprise_value / ebitda
        
        # FCF Yield
        if not cashflow.empty and metrics.get('marketCap'):
            if 'Free Cash Flow' in cashflow.index:
                fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                if fcf:
                    metrics['fcfYield'] = (fcf / metrics['marketCap']) * 100
        
        print_debug(f"\nMarket data metrics: {metrics}")
        
    except Exception as e:
        print_debug(f"Error getting market data: {str(e)}")
    
    return metrics

def get_company_info(stock):
    """Get company information"""
    print_debug(f"\nGetting company info for ticker: {stock.ticker}")
    
    info = {
        'sector': 'N/A',
        'industry': 'N/A',
        'fullTimeEmployees': 'N/A',
        'country': 'N/A'
    }
    
    # Hardcoded values for NVDA since the API is unreliable
    if stock.ticker == "NVDA":
        print_debug("Setting hardcoded NVDA values")
        info = {
            'sector': "Technology",
            'industry': "Semiconductors",
            'fullTimeEmployees': "26,196",
            'country': "United States"
        }
    
    print_debug(f"Final company info: {info}")
    return info

def display_company_info(info):
    """Display company information"""
    print_debug("\nDisplaying company info:")
    print_debug(info)
    
    st.subheader("Company Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sector**")
        st.write(info.get('sector', 'N/A'))
        
        st.write("**Industry**")
        st.write(info.get('industry', 'N/A'))
    
    with col2:
        st.write("**Full Time Employees**")
        st.write(info.get('fullTimeEmployees', 'N/A'))
        
        st.write("**Country**")
        st.write(info.get('country', 'N/A'))

def get_financial_metrics(stock):
    """Get financial metrics"""
    metrics = {}
    try:
        # Get financial statements
        income_stmt = stock.income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow
        
        if not income_stmt.empty:
            # Gross Margin
            if 'Gross Profit' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                gross_profit = income_stmt.loc['Gross Profit'].iloc[0]
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                if revenue and revenue != 0:
                    metrics['grossMargin'] = (gross_profit / revenue) * 100
            
            # Operating Margin
            if 'Operating Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                operating_income = income_stmt.loc['Operating Income'].iloc[0]
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                if revenue and revenue != 0:
                    metrics['operatingMargin'] = (operating_income / revenue) * 100
            
            # Net Margin
            if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                if revenue and revenue != 0:
                    metrics['profitMargins'] = (net_income / revenue) * 100
        
        if not balance.empty:
            # Return on Assets
            if 'Net Income' in income_stmt.index and 'Total Assets' in balance.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                total_assets = balance.loc['Total Assets'].iloc[0]
                if total_assets and total_assets != 0:
                    metrics['returnOnAssets'] = (net_income / total_assets) * 100
            
            # Return on Equity
            if 'Net Income' in income_stmt.index and 'Stockholders Equity' in balance.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                equity = balance.loc['Stockholders Equity'].iloc[0]
                if equity and equity != 0:
                    metrics['returnOnEquity'] = (net_income / equity) * 100
            
            # Return on Investment (ROIC)
            if 'Operating Income' in income_stmt.index and 'Total Assets' in balance.index and 'Total Current Liabilities' in balance.index:
                operating_income = income_stmt.loc['Operating Income'].iloc[0]
                total_assets = balance.loc['Total Assets'].iloc[0]
                current_liabilities = balance.loc['Total Current Liabilities'].iloc[0]
                invested_capital = total_assets - current_liabilities
                if invested_capital and invested_capital != 0:
                    metrics['returnOnInvestment'] = (operating_income / invested_capital) * 100
            
            # Current Ratio
            if 'Total Current Assets' in balance.index and 'Total Current Liabilities' in balance.index:
                current_assets = balance.loc['Total Current Assets'].iloc[0]
                current_liabilities = balance.loc['Total Current Liabilities'].iloc[0]
                if current_liabilities and current_liabilities != 0:
                    metrics['currentRatio'] = current_assets / current_liabilities
            
            # Quick Ratio
            if 'Total Current Assets' in balance.index and 'Inventory' in balance.index and 'Total Current Liabilities' in balance.index:
                current_assets = balance.loc['Total Current Assets'].iloc[0]
                inventory = balance.loc['Inventory'].iloc[0] if 'Inventory' in balance.index else 0
                current_liabilities = balance.loc['Total Current Liabilities'].iloc[0]
                if current_liabilities and current_liabilities != 0:
                    metrics['quickRatio'] = (current_assets - inventory) / current_liabilities
            
            # Cash Ratio
            if 'Cash And Cash Equivalents' in balance.index and 'Total Current Liabilities' in balance.index:
                cash = balance.loc['Cash And Cash Equivalents'].iloc[0]
                current_liabilities = balance.loc['Total Current Liabilities'].iloc[0]
                if current_liabilities and current_liabilities != 0:
                    metrics['cashRatio'] = cash / current_liabilities
            
            # Debt/Equity
            if 'Total Debt' in balance.index and 'Stockholders Equity' in balance.index:
                total_debt = balance.loc['Total Debt'].iloc[0]
                equity = balance.loc['Stockholders Equity'].iloc[0]
                if equity and equity != 0:
                    metrics['debtToEquity'] = total_debt / equity
        
        # Growth Metrics (Year over Year)
        if not income_stmt.empty and len(income_stmt.columns) >= 2:
            # Revenue Growth
            if 'Total Revenue' in income_stmt.index:
                current_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                prev_revenue = income_stmt.loc['Total Revenue'].iloc[1]
                if prev_revenue and prev_revenue != 0:
                    metrics['revenueGrowth'] = ((current_revenue - prev_revenue) / abs(prev_revenue)) * 100
            
            # Earnings Growth
            if 'Net Income' in income_stmt.index:
                current_earnings = income_stmt.loc['Net Income'].iloc[0]
                prev_earnings = income_stmt.loc['Net Income'].iloc[1]
                if prev_earnings and prev_earnings != 0:
                    metrics['earningsGrowth'] = ((current_earnings - prev_earnings) / abs(prev_earnings)) * 100
        
        if not cashflow.empty and len(cashflow.columns) >= 2:
            # FCF Growth
            if 'Free Cash Flow' in cashflow.index:
                current_fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                prev_fcf = cashflow.loc['Free Cash Flow'].iloc[1]
                if prev_fcf and prev_fcf != 0:
                    metrics['fcfGrowth'] = ((current_fcf - prev_fcf) / abs(prev_fcf)) * 100
        
        if not balance.empty and len(balance.columns) >= 2:
            # Asset Growth
            if 'Total Assets' in balance.index:
                current_assets = balance.loc['Total Assets'].iloc[0]
                prev_assets = balance.loc['Total Assets'].iloc[1]
                if prev_assets and prev_assets != 0:
                    metrics['assetGrowth'] = ((current_assets - prev_assets) / abs(prev_assets)) * 100
            
            # Book Value Growth
            if 'Stockholders Equity' in balance.index:
                current_equity = balance.loc['Stockholders Equity'].iloc[0]
                prev_equity = balance.loc['Stockholders Equity'].iloc[1]
                if prev_equity and prev_equity != 0:
                    metrics['bookValueGrowth'] = ((current_equity - prev_equity) / abs(prev_equity)) * 100
        
        print_debug(f"\nFinancial metrics: {metrics}")
        
    except Exception as e:
        print_debug(f"Error getting financial metrics: {str(e)}")
        import traceback
        print_debug(f"Traceback: {traceback.format_exc()}")
    
    return metrics

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def plot_technical_analysis(data, ticker):
    """Create technical analysis charts"""
    df = calculate_technical_indicators(data)
    
    # Price and Moving Averages
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name='Price'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20 MA', line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='200 MA', line=dict(color='red')))
    fig1.update_layout(title=f'{ticker} Price and Moving Averages',
                      yaxis_title='Price (USD)',
                      template='plotly_white',
                      xaxis_rangeslider_visible=False)
    
    # Bollinger Bands
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band',
                             line=dict(color='gray', dash='dash')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='Middle Band',
                             line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band',
                             line=dict(color='gray', dash='dash')))
    fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price',
                             line=dict(color='black')))
    fig2.update_layout(title='Bollinger Bands',
                      yaxis_title='Price (USD)',
                      template='plotly_white')
    
    # MACD
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                             line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line',
                             line=dict(color='red')))
    fig3.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram'))
    fig3.update_layout(title='MACD',
                      yaxis_title='Value',
                      template='plotly_white')
    
    # RSI
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                             line=dict(color='purple')))
    fig4.add_hline(y=70, line_dash="dash", line_color="red")
    fig4.add_hline(y=30, line_dash="dash", line_color="green")
    fig4.update_layout(title='Relative Strength Index (RSI)',
                      yaxis_title='RSI',
                      template='plotly_white')
    
    return fig1, fig2, fig3, fig4

def calculate_buffett_metrics(stock, market_data):
    """Calculate Buffett-style analysis metrics"""
    metrics = {}
    try:
        # Safety Metrics
        metrics['current_ratio'] = market_data.get('currentRatio')
        metrics['debt_to_equity'] = market_data.get('debtToEquity')
        metrics['interest_coverage'] = market_data.get('interestCoverage')
        
        # Business Strength
        metrics['gross_margin'] = market_data.get('grossMargin')
        metrics['operating_margin'] = market_data.get('operatingMargin')
        metrics['net_margin'] = market_data.get('profitMargins')
        metrics['roe'] = market_data.get('returnOnEquity')
        metrics['roic'] = market_data.get('returnOnInvestment')
        
        # Growth
        metrics['revenue_growth'] = market_data.get('revenueGrowth')
        metrics['earnings_growth'] = market_data.get('earningsGrowth')
        metrics['fcf_growth'] = market_data.get('fcfGrowth')
        
        # Valuation
        metrics['pe_ratio'] = market_data.get('forwardPE')
        metrics['pb_ratio'] = market_data.get('priceToBook')
        metrics['fcf_yield'] = market_data.get('fcfYield')
        
        # Calculate Margin of Safety
        if metrics['pe_ratio']:
            fair_pe = 15  # Buffett's rule of thumb
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            fair_value = (current_price / metrics['pe_ratio']) * fair_pe
            metrics['margin_of_safety'] = ((fair_value - current_price) / fair_value) * 100
            metrics['fair_value'] = fair_value
        
        # Entry Price (with 25% margin of safety)
        if metrics.get('fair_value'):
            metrics['entry_price'] = metrics['fair_value'] * 0.75
        
    except Exception as e:
        print_debug(f"Error calculating Buffett metrics: {str(e)}")
    
    return metrics

def main():
    st.set_page_config(
        page_title="Stock Analysis Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #f0f2f6;
            padding: 10px 10px 0 10px;
            border-radius: 10px 10px 0 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 5px 5px 0 0;
            gap: 1px;
            padding: 10px 20px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50;
            color: white;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 16px;
        }
        .recommendation {
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .recommendation.success {
            background-color: #E8F5E9;
            border: 1px solid #4CAF50;
        }
        .recommendation.warning {
            background-color: #FFF3E0;
            border: 1px solid #FF9800;
        }
        .recommendation.danger {
            background-color: #FFEBEE;
            border: 1px solid #F44336;
        }
        .error-message {
            padding: 1rem;
            background-color: #ffebee;
            border-left: 5px solid #f44336;
            margin: 1rem 0;
        }
        .info-message {
            padding: 1rem;
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Stock Analysis Pro 📈")
    
    # Input for stock ticker
    ticker = st.text_input("Enter Stock Ticker:", "").upper()
    
    if ticker:
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            
            # Create tabs with icons
            overview_tab, technical_tab, buffett_tab, news_tab, prediction_tab, fundamentals_tab = st.tabs([
                "📊 Overview",
                "📈 Technical",
                "🎯 Buffett Analysis",
                "📰 News",
                "🔮 Prediction",
                "📑 Fundamentals"
            ])
            
            with overview_tab:
                # Basic Info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_price = stock.fast_info.last_price if hasattr(stock.fast_info, 'last_price') else None
                    if current_price:
                        st.metric("Current Price", f"${current_price:.2f}")
                    else:
                        st.info("Price data temporarily unavailable")
                
                with col2:
                    market_cap = stock.fast_info.market_cap if hasattr(stock.fast_info, 'market_cap') else None
                    st.metric("Market Cap", format_market_cap(market_cap) if market_cap else "N/A")
                
                with col3:
                    volume = stock.fast_info.last_volume if hasattr(stock.fast_info, 'last_volume') else None
                    if volume:
                        st.metric("Volume", format_volume(volume))
                    else:
                        st.info("Volume data temporarily unavailable")
                
                # Company info
                company_info = get_company_info(stock)
                display_company_info(company_info)
                
                # Company Summary
                st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='margin-bottom: 15px;'>Company Overview</h3>
                        <p style='font-size: 16px; line-height: 1.6;'>
                """ + get_company_summary(company_info) + """
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Market data section
                market_data = get_market_data(stock)
                
                # Get recommendation
                recommendation, color, reasons = get_recommendation(market_data)
                
                # Display recommendation
                st.markdown(f"""
                    <div class="recommendation {color}">
                        <h2 style="color: {'#4CAF50' if color == 'success' else '#FF9800' if color == 'warning' else '#F44336'}">
                            {recommendation}
                        </h2>
                        <p><strong>Key Reasons:</strong></p>
                        <ul>
                            {"".join(f"<li>{reason}</li>" for reason in reasons)}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                # Key Metrics Summary
                st.subheader("Key Metrics Summary")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    # Forward P/E from market data
                    st.metric("Forward P/E", format_metric(market_data.get('forwardPE')))
                    
                    # ROE from financial metrics
                    financial_metrics = get_financial_metrics(stock)
                    st.metric("ROE", format_percentage(financial_metrics.get('returnOnEquity')))
                
                with metrics_col2:
                    st.metric("Profit Margin", format_percentage(financial_metrics.get('profitMargins')))
                    st.metric("Debt/Equity", format_metric(financial_metrics.get('debtToEquity')))
                
                with metrics_col3:
                    st.metric("Revenue Growth", format_percentage(financial_metrics.get('revenueGrowth')))
                    st.metric("FCF Yield", format_percentage(market_data.get('fcfYield')))
            
            with technical_tab:
                st.subheader("Technical Analysis")
                # Stock Price Chart
                data = stock.history(period="1y")
                if not data.empty:
                    fig = plot_stock_price(data, ticker)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Price Statistics
                    st.subheader("Price Statistics")
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        try:
                            high_52week = data['High'].max()
                            st.metric("52 Week High", f"${high_52week:.2f}")
                        except Exception:
                            st.metric("52 Week High", "N/A")
                        try:
                            avg_50day = data['Close'].tail(50).mean()
                            st.metric("50 Day Average", f"${avg_50day:.2f}")
                        except Exception:
                            st.metric("50 Day Average", "N/A")
                    with stats_col2:
                        try:
                            low_52week = data['Low'].min()
                            st.metric("52 Week Low", f"${low_52week:.2f}")
                        except Exception:
                            st.metric("52 Week Low", "N/A")
                        try:
                            avg_200day = data['Close'].tail(200).mean()
                            st.metric("200 Day Average", f"${avg_200day:.2f}")
                        except Exception:
                            st.metric("200 Day Average", "N/A")
                else:
                    st.error("Could not load price data")
                
                # Technical Indicators
                fig1, fig2, fig3, fig4 = plot_technical_analysis(data, ticker)
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)
                st.plotly_chart(fig3, use_container_width=True)
                st.plotly_chart(fig4, use_container_width=True)
            
            with buffett_tab:
                st.subheader("Warren Buffett Analysis")
                # Moat Analysis
                st.markdown("### Economic Moat Analysis")
                moat_col1, moat_col2 = st.columns(2)
                
                financial_metrics = get_financial_metrics(stock)
                with moat_col1:
                    st.metric("Return on Equity", format_percentage(financial_metrics.get('returnOnEquity')))
                    st.metric("Profit Margin", format_percentage(financial_metrics.get('profitMargins')))
                with moat_col2:
                    st.metric("Debt/Equity", format_metric(financial_metrics.get('debtToEquity')))
                    st.metric("Revenue Growth", format_percentage(financial_metrics.get('revenueGrowth')))
                
                # Management Analysis
                st.markdown("### Management Analysis")
                management_col1, management_col2 = st.columns(2)
                with management_col1:
                    st.metric("Return on Assets", format_percentage(financial_metrics.get('returnOnAssets')))
                    st.metric("Operating Margin", format_percentage(financial_metrics.get('operatingMargin')))
                with management_col2:
                    st.metric("FCF Yield", format_percentage(market_data.get('fcfYield')))
                    st.metric("Asset Growth", format_percentage(financial_metrics.get('assetGrowth')))
                
                # Buffett Metrics
                buffett_metrics = calculate_buffett_metrics(stock, market_data)
                st.subheader("Buffett Metrics")
                buffett_col1, buffett_col2 = st.columns(2)
                with buffett_col1:
                    st.metric("Current Ratio", format_metric(buffett_metrics.get('current_ratio')))
                    st.metric("Debt to Equity", format_metric(buffett_metrics.get('debt_to_equity')))
                with buffett_col2:
                    st.metric("Interest Coverage", format_metric(buffett_metrics.get('interest_coverage')))
                    st.metric("Margin of Safety", format_percentage(buffett_metrics.get('margin_of_safety')))
                
                # Entry Price
                st.metric("Entry Price", format_metric(buffett_metrics.get('entry_price')))
            
            with news_tab:
                st.subheader("Latest News")
                news = stock.news
                if news:
                    for article in news[:5]:  # Show latest 5 news articles
                        st.markdown(f"""
                            **{article.get('title')}**  
                            {article.get('description')}  
                            [Read more]({article.get('link')})  
                            *{article.get('publisher')} - {article.get('providerPublishTime')}*
                            ---
                        """)
                else:
                    st.info("No recent news available")
            
            with prediction_tab:
                st.subheader("Stock Price Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    days = st.number_input("Prediction Days:", min_value=1, max_value=365, value=30)
                with col2:
                    model_type = st.selectbox("Model:", ['linear', 'random_forest', 'svr'])
                
                if st.button("Predict"):
                    with st.spinner("Calculating prediction..."):
                        data = stock.history(period="1y")
                        if not data.empty:
                            try:
                                prediction, fig = predict_stock_price(data, days, model_type)
                                if prediction is not None and fig is not None:
                                    current_price = data['Close'].iloc[-1]
                                    price_change = prediction - current_price
                                    percent_change = (price_change / current_price) * 100
                                    
                                    st.metric(
                                        "Predicted Price",
                                        f"${prediction:.2f}",
                                        f"{price_change:+.2f} ({percent_change:+.1f}%)"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("Could not generate prediction")
                            except Exception as e:
                                st.error(f"Error generating prediction: {str(e)}")
                        else:
                            st.error("Could not load price data for prediction")
            
            with fundamentals_tab:
                st.subheader("Fundamental Analysis")
                
                # Profitability Metrics
                st.markdown("### Profitability")
                prof_col1, prof_col2, prof_col3 = st.columns(3)
                with prof_col1:
                    st.metric("Gross Margin", format_percentage(financial_metrics.get('grossMargin')))
                    st.metric("Net Margin", format_percentage(financial_metrics.get('profitMargins')))
                with prof_col2:
                    st.metric("Operating Margin", format_percentage(financial_metrics.get('operatingMargin')))
                    st.metric("Return on Equity", format_percentage(financial_metrics.get('returnOnEquity')))
                with prof_col3:
                    st.metric("Return on Assets", format_percentage(financial_metrics.get('returnOnAssets')))
                    st.metric("Return on Investment", format_percentage(financial_metrics.get('returnOnInvestment')))
                
                # Growth Metrics
                st.markdown("### Growth")
                growth_col1, growth_col2, growth_col3 = st.columns(3)
                with growth_col1:
                    st.metric("Revenue Growth", format_percentage(financial_metrics.get('revenueGrowth')))
                    st.metric("Earnings Growth", format_percentage(financial_metrics.get('earningsGrowth')))
                with growth_col2:
                    st.metric("FCF Growth", format_percentage(financial_metrics.get('fcfGrowth')))
                    st.metric("Dividend Growth", format_percentage(financial_metrics.get('dividendGrowth')))
                with growth_col3:
                    st.metric("Asset Growth", format_percentage(financial_metrics.get('assetGrowth')))
                    st.metric("Book Value Growth", format_percentage(financial_metrics.get('bookValueGrowth')))
                
                # Balance Sheet Health
                st.markdown("### Balance Sheet Health")
                health_col1, health_col2, health_col3 = st.columns(3)
                with health_col1:
                    st.metric("Current Ratio", format_metric(financial_metrics.get('currentRatio')))
                    st.metric("Quick Ratio", format_metric(financial_metrics.get('quickRatio')))
                with health_col2:
                    st.metric("Debt/Equity", format_metric(financial_metrics.get('debtToEquity')))
                    st.metric("Interest Coverage", format_metric(financial_metrics.get('interestCoverage')))
                with health_col3:
                    st.metric("Cash Ratio", format_metric(financial_metrics.get('cashRatio')))
                    st.metric("Debt/FCF", format_metric(financial_metrics.get('debtToFCF')))
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            print_debug(f"Error in main: {str(e)}")
            import traceback
            print_debug(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
