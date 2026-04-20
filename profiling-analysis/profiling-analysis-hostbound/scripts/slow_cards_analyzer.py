import argparse
import os
import sqlite3
import pandas as pd

# 定义分析相关阈值
SPEED_THRESHOLD = 0.2     # 快慢卡判定标准差倍数（默认20%）


def merge_databases(db_paths, output_db_path='merged_analysis.db'):
    """合并多个profiling数据库文件"""
    if not db_paths:
        raise ValueError("至少需要一个数据库文件路径")
    
    # 创建输出数据库
    merged_conn = sqlite3.connect(output_db_path)
    merged_cursor = merged_conn.cursor()
    
    # 创建StepTraceTime表
    merged_cursor.execute('''
    CREATE TABLE IF NOT EXISTS StepTraceTime (
        deviceId INTEGER,
        step TEXT,
        computing NUMERIC,
        communication_not_overlapped NUMERIC,
        overlapped NUMERIC,
        communication NUMERIC,
        free NUMERIC,
        stage NUMERIC,
        bubble NUMERIC,
        communication_not_overlapped_and_exclude_receive NUMERIC,
        preparing NUMERIC
    )
    ''')
    
    # 合并所有数据库
    for db_path in db_paths:
        if not os.path.exists(db_path):
            print(f"警告: 数据库文件不存在 {db_path}")
            continue
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM StepTraceTime", conn)
        conn.close()
        
        if not df.empty:
            df.to_sql('StepTraceTime', merged_conn, if_exists='append', index=False)
    
    # 验证合并结果
    merged_df = pd.read_sql_query("SELECT * FROM StepTraceTime", merged_conn)
    merged_conn.close()
    
    print(f"\n=== 数据库合并结果 ===")
    print(f"- 合并了 {len(db_paths)} 个数据库文件")
    print(f"- 合并后设备数量: {merged_df['deviceId'].nunique()}")
    print(f"- 合并后设备列表: {merged_df['deviceId'].unique().tolist()}")
    
    return output_db_path


class MindStudioProfilerAnalyzer:
    def __init__(self, db_path, speed_threshold=SPEED_THRESHOLD):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.speed_threshold = speed_threshold

    def connect(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Profiling数据库文件不存在: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.close()

    def analyze_card_performance(self):
        """分析卡的性能表现，识别快慢卡"""
        try:
            # 查询StepTraceTime表获取设备级性能数据
            self.cursor.execute("SELECT deviceId, computing, communication FROM StepTraceTime")
            task_data = self.cursor.fetchall()
            if not task_data:
                return None, "未找到设备级性能数据"

            df_tasks = pd.DataFrame(task_data, columns=['device_id', 'computing', 'communication'])
            
            # 按设备分组统计
            card_metrics = df_tasks.groupby('device_id').agg({
                'computing': ['mean', 'std', 'count'],
                'communication': ['mean', 'std']
            }).reset_index()
            
            # 展平列名
            card_metrics.columns = ['device_id', 'avg_computing', 'std_computing', 'task_count', 'avg_communication', 'std_communication']
            
            # 识别快慢卡（综合计算时间和通信时间）
            overall_avg_computing = card_metrics['avg_computing'].mean()
            overall_std_computing = card_metrics['avg_computing'].std()
            
            # 计算通信时间的统计信息
            overall_avg_communication = card_metrics['avg_communication'].mean()
            overall_std_communication = card_metrics['avg_communication'].std()
            
            # 只有一个设备时，无法比较快慢
            if len(card_metrics) <= 1:
                slow_cards_computing = []
                fast_cards_computing = []
                slow_cards_communication = []
                fast_cards_communication = []
                normal_cards = card_metrics['device_id'].tolist() if not card_metrics.empty else []
            else:
                # 计算时间的快慢卡阈值
                slow_threshold_computing = overall_avg_computing + overall_std_computing * self.speed_threshold
                fast_threshold_computing = overall_avg_computing - overall_std_computing * self.speed_threshold
                
                # 通信时间的快慢卡阈值（注意：通信时间越短可能表示等待时间越短，性能越好）
                slow_threshold_communication = overall_avg_communication + overall_std_communication * self.speed_threshold
                fast_threshold_communication = overall_avg_communication - overall_std_communication * self.speed_threshold
                
                # 计算时间维度的快慢卡
                slow_cards_computing = card_metrics[card_metrics['avg_computing'] > slow_threshold_computing]['device_id'].tolist()
                fast_cards_computing = card_metrics[card_metrics['avg_computing'] < fast_threshold_computing]['device_id'].tolist()
                
                # 通信时间维度的快慢卡（根据用户反馈修正：通信时间长说明该卡计算快，在等待慢卡，所以是快卡）
                # 通信时间长 = 快卡（等待其他卡）
                # 通信时间短 = 慢卡（被其他卡等待）
                fast_cards_communication = card_metrics[card_metrics['avg_communication'] > slow_threshold_communication]['device_id'].tolist()
                slow_cards_communication = card_metrics[card_metrics['avg_communication'] < fast_threshold_communication]['device_id'].tolist()
                
                # 正常卡（在两个维度都正常的卡）
                normal_cards = card_metrics[
                    (card_metrics['avg_computing'] <= slow_threshold_computing) & 
                    (card_metrics['avg_computing'] >= fast_threshold_computing) &
                    (card_metrics['avg_communication'] <= slow_threshold_communication) &
                    (card_metrics['avg_communication'] >= fast_threshold_communication)
                ]['device_id'].tolist()
            
            # 综合判定：只要在任一维度表现异常，就认为是快慢卡
            all_slow_cards = list(set(slow_cards_computing + slow_cards_communication))
            all_fast_cards = list(set(fast_cards_computing + fast_cards_communication))
            
            return {
                'card_metrics': card_metrics.to_dict('records'),
                'slow_cards': all_slow_cards,
                'fast_cards': all_fast_cards,
                'normal_cards': normal_cards,
                'slow_cards_computing': slow_cards_computing,
                'fast_cards_computing': fast_cards_computing,
                'slow_cards_communication': slow_cards_communication,
                'fast_cards_communication': fast_cards_communication,
                'overall_avg_computing': overall_avg_computing,
                'overall_std_computing': overall_std_computing,
                'overall_avg_communication': overall_avg_communication,
                'overall_std_communication': overall_std_communication,
                'has_speed_imbalance': len(all_slow_cards) > 0 and len(all_fast_cards) > 0
            }, None
            
        except Exception as e:
            return None, f"卡性能分析失败: {str(e)}"

    def generate_report(self, card_analysis, output_path):
        """生成性能分析报告"""
        if not card_analysis:
            return output_path
        
        # 创建分析摘要
        report_data = {
            '分析项': [
                '是否存在快慢卡不均衡',
                '综合慢卡列表',
                '综合快卡列表',
                '计算时间慢卡',
                '计算时间快卡',
                '通信时间慢卡',
                '通信时间快卡',
                '卡性能整体平均计算时间',
                '卡性能计算时间标准差',
                '卡性能整体平均通信时间',
                '卡性能通信时间标准差'
            ],
            '数值': [
                '是' if card_analysis['has_speed_imbalance'] else '否',
                ', '.join(map(str, card_analysis['slow_cards'])) if card_analysis['slow_cards'] else '无',
                ', '.join(map(str, card_analysis['fast_cards'])) if card_analysis['fast_cards'] else '无',
                ', '.join(map(str, card_analysis['slow_cards_computing'])) if card_analysis['slow_cards_computing'] else '无',
                ', '.join(map(str, card_analysis['fast_cards_computing'])) if card_analysis['fast_cards_computing'] else '无',
                ', '.join(map(str, card_analysis['slow_cards_communication'])) if card_analysis['slow_cards_communication'] else '无',
                ', '.join(map(str, card_analysis['fast_cards_communication'])) if card_analysis['fast_cards_communication'] else '无',
                f"{card_analysis['overall_avg_computing']:.2f}",
                f"{card_analysis['overall_std_computing']:.2f}",
                f"{card_analysis['overall_avg_communication']:.2f}",
                f"{card_analysis['overall_std_communication']:.2f}"
            ],
            '状态': [
                '异常' if card_analysis['has_speed_imbalance'] else '正常',
                '正常' if not card_analysis['slow_cards'] else '异常',
                '正常' if not card_analysis['fast_cards'] else '异常',
                '正常' if not card_analysis['slow_cards_computing'] else '异常',
                '正常' if not card_analysis['fast_cards_computing'] else '异常',
                '正常' if not card_analysis['slow_cards_communication'] else '异常',
                '正常' if not card_analysis['fast_cards_communication'] else '异常',
                '正常',
                '正常',
                '正常',
                '正常'
            ]
        }

        df_report = pd.DataFrame(report_data)

        # 保存报告
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_report.to_excel(writer, sheet_name='分析摘要', index=False)
            
            # 卡性能详细数据
            pd.DataFrame(card_analysis['card_metrics']).to_excel(writer, sheet_name='卡性能详情', index=False)

        return output_path


def main():
    parser = argparse.ArgumentParser(description='MindStudio Insight Profiling 快慢卡分析工具')
    parser.add_argument('--db-paths', type=str, nargs='+', required=True, help='Profiling数据库文件路径列表(analysis.db)')
    parser.add_argument('--output', type=str, default='card_performance_analysis.xlsx', help='分析报告输出路径')
    parser.add_argument('--speed-threshold', type=float, default=SPEED_THRESHOLD, help=f'快慢卡判定标准差倍数(默认: {SPEED_THRESHOLD})')
    args = parser.parse_args()

    merged_db_path = None

    # 合并数据库
    if len(args.db_paths) > 1:
        merged_db_path = merge_databases(args.db_paths)
        db_path_to_use = merged_db_path
    else:
        db_path_to_use = args.db_paths[0]

    analyzer = MindStudioProfilerAnalyzer(db_path_to_use, args.speed_threshold)

    try:
        analyzer.connect()

        print("开始分析下发问题与快慢卡性能...")

        # 分析性能
        card_analysis, card_error = analyzer.analyze_card_performance()
        if card_error:
            print(f"性能分析警告: {card_error}")

        # 生成报告
        report_path = analyzer.generate_report(card_analysis, args.output)
        print(f"分析报告已生成: {report_path}")

        # 分析结果输出
        if card_analysis:
            print("\n=== 性能分析结果 ===")
            
            # 综合快慢卡分析
            if card_analysis['has_speed_imbalance']:
                print("检测到快慢卡性能不均衡：")
                print(f"- 【综合慢卡】({len(card_analysis['slow_cards'])}个): {', '.join(map(str, card_analysis['slow_cards']))}")
                print(f"  这些卡的性能在计算或通信维度表现较差，影响整体集群性能")
                print(f"- 综合快卡({len(card_analysis['fast_cards'])}个): {', '.join(map(str, card_analysis['fast_cards']))}")
                print(f"- 正常卡({len(card_analysis['normal_cards'])}个): {', '.join(map(str, card_analysis['normal_cards']))}")
            else:
                print("未检测到明显的快慢卡性能不均衡问题")
            
            # 计算时间维度分析
            print(f"\n=== 计算时间维度分析 ===")
            print(f"- 整体平均计算时间: {card_analysis['overall_avg_computing']:.2f}ms")
            print(f"- 计算时间标准差: {card_analysis['overall_std_computing']:.2f}ms")
            if card_analysis['slow_cards_computing']:
                print(f"- 【计算时间慢卡】({len(card_analysis['slow_cards_computing'])}个): {', '.join(map(str, card_analysis['slow_cards_computing']))}")
                print(f"  这些卡的计算时间明显长于平均水平")
            else:
                print("- 无计算时间慢卡")
            if card_analysis['fast_cards_computing']:
                print(f"- 计算时间快卡({len(card_analysis['fast_cards_computing'])}个): {', '.join(map(str, card_analysis['fast_cards_computing']))}")
            else:
                print("- 无计算时间快卡")
            
            # 通信时间维度分析
            print(f"\n=== 通信时间维度分析 ===")
            print(f"- 整体平均通信时间: {card_analysis['overall_avg_communication']:.2f}ms")
            print(f"- 通信时间标准差: {card_analysis['overall_std_communication']:.2f}ms")
            if card_analysis['slow_cards_communication']:
                print(f"- 【通信时间慢卡】({len(card_analysis['slow_cards_communication'])}个): {', '.join(map(str, card_analysis['slow_cards_communication']))}")
                print("  说明: 通信时间短表示该卡计算慢，被其他卡等待")
                print("  注意: 通信时间短并不一定是通信量导致的，该skill属于hostbound分支")
                print("  提示: 由于该卡处于free占比高的状态，快慢卡可能是由于下发问题导致的")
            else:
                print("- 无通信时间慢卡")
            if card_analysis['fast_cards_communication']:
                print(f"- 通信时间快卡({len(card_analysis['fast_cards_communication'])}个): {', '.join(map(str, card_analysis['fast_cards_communication']))}")
                print("  说明: 通信时间长表示该卡计算快，在等待其他卡完成计算")
            else:
                print("- 无通信时间快卡")
        
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
    finally:
        if 'analyzer' in locals():
            analyzer.close()
        # 清理临时合并文件
        if merged_db_path and os.path.exists(merged_db_path):
            try:
                os.remove(merged_db_path)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    main()
