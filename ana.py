import pandas as pd

from utils.judgement_utils import judge_ai_match, normalize_final_result


def analyse_result(result_file, output_file):
	df =pd.read_excel(result_file, sheet_name="Sheet1")


	# 归一化 AI 和 vendor judgement
	df["ai_norm"] = df["final_result"].apply(normalize_final_result)
	df["vendor_norm"] = df["vendor judgement"].apply(normalize_final_result)

	# match: True/False 表示可比且一致；None 表示不可比(未知值)
	df["match"] = df.apply(
		lambda r: judge_ai_match(r.get("ai_norm"), r.get("vendor_norm"), ignore=set()),
		axis=1,
	)

	# 将布尔/None 的匹配结果映射为符号：√ = 一致, × = 不一致, - = 不可比
	df["match_symbol"] = df["match"].map({True: "√", False: "×"}).fillna("-")

	# 四类标签结构：Correct / Incorrect / NeedDiscussion / Spam / Unknown
	label_order = ["Correct", "Incorrect", "NeedDiscussion", "Spam", "Unknown"]
	df["ai_label4"] = df["ai_norm"].fillna("Unknown")
	df["vendor_label4"] = df["vendor_norm"].fillna("Unknown")

	# 整体混淆矩阵（AI 预测在行，vendor 标注在列）
	confusion = (
		pd.crosstab(
			df["ai_label4"],
			df["vendor_label4"],
			rownames=["AI"],
			colnames=["Vendor"],
			dropna=False,
		)
		.reindex(index=label_order, columns=label_order, fill_value=0)
	)


	def _metrics(group: pd.DataFrame) -> pd.Series:
		total = len(group)
		comparable = group["match"].notna().sum()
		correct = (group["match"] == True).sum()  # noqa: E712
		acc = (correct / comparable * 100.0) if comparable else float("nan")
		coverage = (comparable / total * 100.0) if total else float("nan")
		need_discussion = (group["ai_norm"] == "NeedDiscussion").sum()
		spam = (group["ai_norm"] == "Spam").sum()
		unknown_ai = group["ai_norm"].isna().sum()
		unknown_vendor = group["vendor_norm"].isna().sum()

		return pd.Series(
			{
				"total": total,
				"comparable": comparable,
				"accuracy_%": round(acc, 2) if acc == acc else acc,  # keep NaN
				"coverage_%": round(coverage, 2) if coverage == coverage else coverage,
				"ai_need_discussion": int(need_discussion),
				"ai_spam": int(spam),
				"ai_unknown": int(unknown_ai),
				"vendor_unknown": int(unknown_vendor),
			}
		)


	# 按 '类型' 分组统计准确度指标（兼容不同 pandas 版本）
	try:
		by_type = df.groupby("类型", dropna=False).apply(_metrics, include_groups=False)
	except TypeError:
		by_type = df.groupby("类型", dropna=False).apply(_metrics)

	by_type = by_type.sort_values(by=["coverage_%", "accuracy_%"], ascending=False)

	# 每个类型下，AI 和 vendor 的四类分布
	ai_by_type = (
		df.pivot_table(
			index="类型",
			columns="ai_label4",
			values="final_result",
			aggfunc="count",
			fill_value=0,
		)
		.reindex(columns=label_order, fill_value=0)
	)

	vendor_by_type = (
		df.pivot_table(
			index="类型",
			columns="vendor_label4",
			values="final_result",
			aggfunc="count",
			fill_value=0,
		)
		.reindex(columns=label_order, fill_value=0)
	)

	overall_total = len(df)
	overall_comparable = df["match"].notna().sum()
	overall_correct = (df["match"] == True).sum()  # noqa: E712
	overall_acc = (overall_correct / overall_comparable * 100.0) if overall_comparable else float("nan")
	overall_coverage = (overall_comparable / overall_total * 100.0) if overall_total else float("nan")

	print("\nOverall:")
	print(
		{
			"total": overall_total,
			"comparable": int(overall_comparable),
			"accuracy_%": round(overall_acc, 2) if overall_acc == overall_acc else overall_acc,
			"coverage_%": round(overall_coverage, 2) if overall_coverage == overall_coverage else overall_coverage,
		}
	)

	print("\n按类型分组统计：")
	print(by_type)

	print("\n四类整体混淆矩阵（AI 行, Vendor 列）：")
	print(confusion)

	print("\n按类型统计 AI 四类分布：")
	print(ai_by_type)

	print("\n按类型统计 Vendor 四类分布：")
	print(vendor_by_type)

	# 将详细数据和四类结构分析一起导出到一个 Excel 文件
	with pd.ExcelWriter(output_file) as writer:
		df.to_excel(writer, sheet_name="data", index=False)
		by_type.to_excel(writer, sheet_name="metrics_by_type")
		confusion.to_excel(writer, sheet_name="confusion_4class")
		ai_by_type.to_excel(writer, sheet_name="ai_4class_by_type")
		vendor_by_type.to_excel(writer, sheet_name="vendor_4class_by_type")

	print(f"\n已导出对比结果到: {output_file}")


def print_output_summary(output_file: str) -> None:
	"""读取 *只有结果* 的 output_file(Excel)，直接计算并打印四类结构和准确率。

	要求 output_file 至少包含列：
	  - final_result: AI 的结果
	  - vendor judgement: vendor 标注
	"""

	try:
		df = pd.read_excel(output_file, sheet_name="Sheet1")
	except FileNotFoundError:
		print(f"output_file 不存在: {output_file}")
		return
	except ValueError:
		# 没有 Sheet1 的话，退回读取默认第一个 sheet
		df = pd.read_excel(output_file)

	missing = [c for c in ["final_result", "vendor judgement"] if c not in df.columns]
	if missing:
		print(f"output_file 中缺少必要列: {missing}")
		return

	# 归一化到四类标签
	df["ai_norm"] = df["final_result"].apply(normalize_final_result)
	df["vendor_norm"] = df["vendor judgement"].apply(normalize_final_result)

	label_order = ["Correct", "Incorrect", "NeedDiscussion", "Spam", "Unknown"]
	df["ai_label4"] = df["ai_norm"].fillna("Unknown")
	df["vendor_label4"] = df["vendor_norm"].fillna("Unknown")

	# 混淆矩阵：Vendor 在行 (纵轴)，AI 在列 (横轴)
	confusion = (
		pd.crosstab(
			df["vendor_label4"],
			df["ai_label4"],
			rownames=["Vendor"],
			colnames=["AI"],
			dropna=False,
		)
		.reindex(index=label_order, columns=label_order, fill_value=0)
	)

	print("\n=== 基于 output_file 计算四类整体混淆矩阵（Vendor 行, AI 列） ===")
	print(confusion)

	# Vendor 视角的分类结果：每行一个 Vendor 类，列为各 AI 预测去向 + 汇总
	conf_vendor_view = confusion.copy()
	conf_vendor_view.columns = [f"{c}_AI" for c in conf_vendor_view.columns]
	conf_vendor_view["Vendor_total"] = confusion.sum(axis=1)
	ai_totals = confusion.sum(axis=0).to_dict()
	conf_vendor_view["AI_total"] = conf_vendor_view.index.map(ai_totals.get)

	print("\n=== 按 Vendor 视角的分类结果（含错分去向） ===")
	print(conf_vendor_view)

	# 将 AI 和 Vendor 的四类分布放在一个维度里对比
	total_all = confusion.to_numpy().sum()
	dist_overall = pd.DataFrame(
		{
			"Vendor_count": confusion.sum(axis=0),
			"AI_count": confusion.sum(axis=1),
		}
	)
	if total_all:
		dist_overall["Vendor_pct"] = dist_overall["Vendor_count"] / total_all * 100.0
		dist_overall["AI_pct"] = dist_overall["AI_count"] / total_all * 100.0

	print("\n=== 四类整体分布（AI 和 Vendor 同行展示） ===")
	print(dist_overall)

	print("\n=== 各类准确率（以 Vendor 为真实标签） ===")
	for label in label_order:
		total_vendor = confusion[label].sum()
		correct = confusion.at[label, label] if label in confusion.index else 0
		acc = (correct / total_vendor * 100.0) if total_vendor else float("nan")
		print(f"{label}: {correct}/{total_vendor} ({acc:.2f}%)")

	# 整体准确率（默认不含 Unknown，聚合所有有效类）
	valid_labels = [lbl for lbl in label_order if lbl != "Unknown"]
	total_vendor_all = confusion.loc[valid_labels].to_numpy().sum()
	correct_all = confusion.loc[valid_labels, valid_labels].to_numpy().trace()
	overall_acc = (correct_all / total_vendor_all * 100.0) if total_vendor_all else float("nan")
	print("\n=== 整体准确率（不含 Unknown） ===")
	print(f"overall: {correct_all}/{total_vendor_all} ({overall_acc:.2f}%)")

	# 如果存在 "类型" 列，则按类型再细分一遍
	if "类型" in df.columns:
		print("\n=== 按 类型 统计四类结构和整体准确率（AI/Vendor 同行展示） ===")
		for type_name, g in df.groupby("类型", dropna=False):
			if g.empty:
				continue

			conf_t = (
				pd.crosstab(
					g["vendor_label4"],
					g["ai_label4"],
					rownames=["Vendor"],
					colnames=["AI"],
					dropna=False,
				)
				.reindex(index=label_order, columns=label_order, fill_value=0)
			)

			# 底部分类展示：将坐标换位，按 AI 行、Vendor 列展示
			conf_t_view = conf_t.T.copy()  # 行：AI，列：Vendor

			print(f"\n--- 类型: {type_name} (AI 行, Vendor 列) ---")
			print(conf_t_view)

			valid_labels_t = [lbl for lbl in label_order if lbl != "Unknown"]
			total_vendor_t = conf_t.loc[valid_labels_t].to_numpy().sum()
			correct_t = conf_t.loc[valid_labels_t, valid_labels_t].to_numpy().trace()
			overall_acc_t = (correct_t / total_vendor_t * 100.0) if total_vendor_t else float("nan")
			print("整体准确率（不含 Unknown）：")
			print(f"overall: {correct_t}/{total_vendor_t} ({overall_acc_t:.2f}%)")



def print_output_field_summary(output_file, comp_field_AI, comp_field_vendor = "vendor judgement"):

	try:
		df = pd.read_excel(output_file, sheet_name="Sheet1")
	except FileNotFoundError:
		print(f"output_file 不存在: {output_file}")
		return
	except ValueError:
		# 没有 Sheet1 的话，退回读取默认第一个 sheet
		df = pd.read_excel(output_file)

	missing = [c for c in [comp_field_AI, comp_field_vendor] if c not in df.columns]
	if missing:
		print(f"output_file 中缺少必要列: {missing}")
		return

	# 归一化到四类标签
	df["ai_norm"] = df[comp_field_AI].apply(normalize_final_result)
	df["vendor_norm"] = df[comp_field_vendor].apply(normalize_final_result)

	label_order = ["Correct", "Incorrect", "NeedDiscussion", "Spam", "Unknown"]
	df["ai_label4"] = df["ai_norm"].fillna("Unknown")
	df["vendor_label4"] = df["vendor_norm"].fillna("Unknown")

	# 混淆矩阵：Vendor 在行 (纵轴)，AI 在列 (横轴)
	confusion = (
		pd.crosstab(
			df["vendor_label4"],
			df["ai_label4"],
			rownames=["Vendor"],
			colnames=["AI"],
			dropna=False,
		)
		.reindex(index=label_order, columns=label_order, fill_value=0)
	)

	print("\n=== 基于 output_file 计算四类整体混淆矩阵（Vendor 行, AI 列） ===")
	print(confusion)

	# Vendor 视角的分类结果：每行一个 Vendor 类，列为各 AI 预测去向 + 汇总
	conf_vendor_view = confusion.copy()
	conf_vendor_view.columns = [f"{c}_AI" for c in conf_vendor_view.columns]
	conf_vendor_view["Vendor_total"] = confusion.sum(axis=1)
	ai_totals = confusion.sum(axis=0).to_dict()
	conf_vendor_view["AI_total"] = conf_vendor_view.index.map(ai_totals.get)

	print("\n=== 按 Vendor 视角的分类结果（含错分去向） ===")
	print(conf_vendor_view)

	# 将 AI 和 Vendor 的四类分布放在一个维度里对比
	total_all = confusion.to_numpy().sum()
	dist_overall = pd.DataFrame(
		{
			"Vendor_count": confusion.sum(axis=0),
			"AI_count": confusion.sum(axis=1),
		}
	)
	if total_all:
		dist_overall["Vendor_pct"] = dist_overall["Vendor_count"] / total_all * 100.0
		dist_overall["AI_pct"] = dist_overall["AI_count"] / total_all * 100.0

	print("\n=== 四类整体分布（AI 和 Vendor 同行展示） ===")
	print(dist_overall)

	print("\n=== 各类准确率（以 Vendor 为真实标签） ===")
	for label in label_order:
		total_vendor = confusion[label].sum()
		correct = confusion.at[label, label] if label in confusion.index else 0
		acc = (correct / total_vendor * 100.0) if total_vendor else float("nan")
		print(f"{label}: {correct}/{total_vendor} ({acc:.2f}%)")

	# 整体准确率（默认不含 Unknown，聚合所有有效类）
	valid_labels = [lbl for lbl in label_order if lbl != "Unknown"]
	total_vendor_all = confusion.loc[valid_labels].to_numpy().sum()
	correct_all = confusion.loc[valid_labels, valid_labels].to_numpy().trace()
	overall_acc = (correct_all / total_vendor_all * 100.0) if total_vendor_all else float("nan")
	print("\n=== 整体准确率（不含 Unknown） ===")
	print(f"overall: {correct_all}/{total_vendor_all} ({overall_acc:.2f}%)")

	# 如果存在 "类型" 列，则按类型再细分一遍
	if "类型" in df.columns:
		print("\n=== 按 类型 统计四类结构和整体准确率（AI/Vendor 同行展示） ===")
		for type_name, g in df.groupby("类型", dropna=False):
			if g.empty:
				continue

			conf_t = (
				pd.crosstab(
					g["vendor_label4"],
					g["ai_label4"],
					rownames=["Vendor"],
					colnames=["AI"],
					dropna=False,
				)
				.reindex(index=label_order, columns=label_order, fill_value=0)
			)

			# 底部分类展示：将坐标换位，按 AI 行、Vendor 列展示
			conf_t_view = conf_t.T.copy()  # 行：AI，列：Vendor

			print(f"\n--- 类型: {type_name} (AI 行, Vendor 列) ---")
			print(conf_t_view)

			valid_labels_t = [lbl for lbl in label_order if lbl != "Unknown"]
			total_vendor_t = conf_t.loc[valid_labels_t].to_numpy().sum()
			correct_t = conf_t.loc[valid_labels_t, valid_labels_t].to_numpy().trace()
			overall_acc_t = (correct_t / total_vendor_t * 100.0) if total_vendor_t else float("nan")
			print("整体准确率（不含 Unknown）：")
			print(f"overall: {correct_t}/{total_vendor_t} ({overall_acc_t:.2f}%)")


if __name__ == "__main__":

	result_file = "test_updated.xlsx"
	output_file = "test_updated_with_match0105.xlsx"
	analyse_result(result_file, output_file)
	# 仅基于 output_file 再计算一次四类准确率并打印
	print_output_field_summary(output_file, "final_result", "vendor judgement")

