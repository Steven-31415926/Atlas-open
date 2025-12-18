import uproot
import awkward as ak
import vector
import traceback
import ROOT
from ROOT import TCanvas, TH1F, TGraphErrors, TF1, TLegend, TLatex, gStyle


# ======== 1. 事件批次处理筛选函数（Data与MC共用） ========
def select_events_by_batch(batch, weights=None, is_mc=False):
    # 光子触发
    trigP = batch["trigP"]
    photon_pt = batch["photon_pt"]
    trigger_mask = (trigP == True)
    two_photon_mask = (ak.num(photon_pt) == 2)
    event_mask = trigger_mask & two_photon_mask

    # 如果没有被触发，返回空数组,后续判定清除
    if ak.sum(event_mask) == 0:
        if weights is not None:
            return ak.Array([]), ak.Array([])
        else:
            return ak.Array([]), None
    
    # 选取被触发的事件转换为GeV单位并进行事件筛选
    pt = photon_pt[event_mask] / 1000.0
    eta = batch["photon_eta"][event_mask]
    phi = batch["photon_phi"][event_mask]
    E = batch["photon_E"][event_mask] / 1000.0
    
    # 读取隔离度
    try:
        ptcone30 = batch["photon_ptcone30"][event_mask]
        etcone20 = batch["photon_etcone20"][event_mask]
    except (KeyError, ValueError):
        ptcone30 = None
        etcone20 = None
    
    # 对MC数据权重处理
    if weights is not None:
        selected_weights = weights[event_mask]
    else:
        selected_weights = None
    
    # 构建四动量并计算不变质量
    photons = vector.zip({"pt": pt, "eta": eta, "phi": phi, "E": E})
    diphoton = photons[:, 0] + photons[:, 1]
    m_gg = diphoton.mass
    
    # 光子筛选
    pt1, pt2 = photons[:, 0].pt, photons[:, 1].pt
    is_swapped = pt1 < pt2
    leading_pt = ak.where(is_swapped, pt2, pt1)
    subleading_pt = ak.where(is_swapped, pt1, pt2)
    leading_et_over_mgg = leading_pt / m_gg
    subleading_et_over_mgg = subleading_pt / m_gg
    
    # cone筛选
    if ptcone30 is not None and etcone20 is not None:
        isolation_mask = (
            (ptcone30[:, 0] < 0.065) & (ptcone30[:, 1] < 0.065) &
            (etcone20[:, 0] < 0.065) & (etcone20[:, 1] < 0.065)
        )
    else:
        print("警告：事件未通过！")
        isolation_mask = ak.zeros_like(leading_pt, dtype=bool)
    
    # 综合筛选
    selection_mask = (
        (leading_pt > 35) &
        (subleading_pt > 25) &
        (leading_et_over_mgg > 0.35) &
        (subleading_et_over_mgg > 0.25) &
        (m_gg > 105) & (m_gg < 160) &
        isolation_mask
    )
    
    # 返回筛选后的m_gg和权重
    if selected_weights is not None:
        return m_gg[selection_mask], selected_weights[selection_mask]
    else:
        return m_gg[selection_mask], None

# ======== 2. 逐个读取Data文件并逐批次处理 ========
data_files = ["data_A.GamGam.root", "data_B.GamGam.root", "data_C.GamGam.root", "data_D.GamGam.root"]

data_m_gg_list = []

print("开始读取Data文件(逐批次处理)...")

# 需要读取的变量列表
data_branches = ["trigP", "photon_pt", "photon_eta", "photon_phi", 
                 "photon_E", "photon_ptcone30", "photon_etcone20"]
#单批次读取大小
size=input("Data文件批处理大小（格式：**MB)")
#分批次读取
for file_name in data_files:
    file_events = 0
    try:
        with uproot.open(file_name) as file:
            tree = file["mini"]
            # 使用 iterate 逐批次读取（每批10MB）
            for batch in tree.iterate(filter_name=data_branches, step_size=size):
                m_gg_batch, _ = select_events_by_batch(batch, weights=None, is_mc=False)
                # 累积结果（只在有有效数据时添加）
                if len(m_gg_batch) > 0:
                    data_m_gg_list.append(m_gg_batch)
                    file_events += len(m_gg_batch)
                # 批次处理完释放内存
                del batch
            
        print(f"  {file_name}: {file_events} 事件通过筛选")      
    except Exception as e:
        print(f"✗ {file_name}: {e}")
        traceback.print_exc() 
        exit(1)

# 合并所有批次
if data_m_gg_list:
    m_gg_data = ak.concatenate(data_m_gg_list)
else:
    m_gg_data = ak.Array([])
    print("没有Data事件通过筛选")

print(f"\nData筛选处理完成,总通过事件数: {len(m_gg_data)}")

# ======== 3. 读取MC文件并逐批次处理 ========
print("\n开始读取MC文件（逐批次处理）...")

#单批次读取大小
size=input("MC文件批处理大小（格式：**MB)")

mc_branches = ["trigP", "photon_pt", "photon_eta", "photon_phi", 
               "photon_E", "photon_ptcone30", "photon_etcone20",
               "mcWeight", "scaleFactor_PHOTON", "scaleFactor_PhotonTRIGGER", 
               "scaleFactor_PILEUP"]
mc_m_gg_list = []
mc_weights_list = []

try:
    with uproot.open("mc_343981.ggH125_gamgam.GamGam.root") as mc_file:
        mc_tree = mc_file["mini"]
        # 逐批次读取MC数据
        for batch in mc_tree.iterate(filter_name=mc_branches, step_size=size):
            # 计算权重
            mcWeight = batch["mcWeight"]
            scaleFactor_PHOTON = batch["scaleFactor_PHOTON"]
            scaleFactor_PhotonTRIGGER = batch["scaleFactor_PhotonTRIGGER"]
            scaleFactor_PILEUP = batch["scaleFactor_PILEUP"]
            mc_total_weight = (mcWeight * scaleFactor_PHOTON * 
                               scaleFactor_PhotonTRIGGER * scaleFactor_PILEUP)
            
            # 处理批次
            m_gg_batch, weights_batch = select_events_by_batch(
                batch, weights=mc_total_weight, is_mc=True
            )
            # 累积结果
            if len(m_gg_batch) > 0:
                mc_m_gg_list.append(m_gg_batch)
                mc_weights_list.append(weights_batch)
            # 释放
            del batch, mc_total_weight
        # 合并
        if mc_m_gg_list:
            m_gg_mc = ak.concatenate(mc_m_gg_list)
            mc_weights_selected = ak.concatenate(mc_weights_list)
        else:
            m_gg_mc = ak.Array([])
            mc_weights_selected = ak.Array([])
            print("警告：没有MC事件通过筛选")
        
        print(f"  MC通过筛选: {len(m_gg_mc)} 事件")
        
except Exception as e:
    print(f"MC文件处理失败: {e}")
    traceback.print_exc() 
    exit(1)

print("\nMC处理完成")


# ======== 4. 预拟合MC信号参数 ========
print("\n预拟合MC信号获取高斯参数...")
if len(m_gg_mc) > 0:
    # 创建MC直方图 填充数据
    h_mc = TH1F("h_mc", "MC Signal", 30, 105, 160)
    h_mc.Sumw2()  # 启用权重误差计算
    for val, w in zip(ak.to_numpy(m_gg_mc), ak.to_numpy(mc_weights_selected)):
        h_mc.Fill(val, w)
    
    # 定义高斯函数并拟合
    f_gauss = TF1("f_gauss", "gaus", 105, 160)
    f_gauss.SetParameters(100, 125.07, 2.51) 
    h_mc.Fit(f_gauss, "RN")
    
    # 拟合参数
    mc_mean = f_gauss.GetParameter(1)
    mc_sigma = f_gauss.GetParameter(2)
    print(f"MC高斯拟合: mean = {mc_mean:.2f} GeV, sigma = {mc_sigma:.2f} GeV")
else:
    mc_mean, mc_sigma = 125.0, 2.51
    print("没有MC事件，使用预设参量，数据需要谨慎对待！")

# ======== 5. 创建Data直方图 ========
if len(m_gg_data) > 0:
    h_data = TH1F("h_data", "Data", 30, 105, 160)
    for val in ak.to_numpy(m_gg_data):
        h_data.Fill(val)
    
    # 误差棒
    g_data = TGraphErrors(h_data)
    g_data.SetMarkerStyle(20)  # 圆点
    g_data.SetMarkerSize(1)
    g_data.SetLineColor(ROOT.kBlack)
else:
    print("没有Data事件！")
    exit(1)

# ======== 6. 联合拟合 ========
# 创建总拟合函数：pol3 + gaus
f_total = TF1("f_total", "pol3(0) + gaus(4)", 105, 160)

# 设置初始参数
f_total.SetParameters(20455.4, -391.151, 2.57413, -0.00576837,
                      h_data.GetMaximum()*0.1, mc_mean, mc_sigma)  

# 设置参数名称和限制
f_total.SetParNames("a", "b", "c", "d", "Signal", "Mean", "Sigma")
f_total.FixParameter(5, mc_mean)   # 固定均值
f_total.FixParameter(6, mc_sigma)  # 固定宽度

# 执行拟合
h_data.Fit(f_total, "RN")
signal_strength = f_total.GetParameter(4)
signal_error = f_total.GetParError(4)

# ======== 8. ROOT 绘图 ========
c = TCanvas("c", "双光子不变质量", 800, 600)

# 绘制数据点
g_data.Draw("AP") 
g_data.SetTitle(";m_{#gamma#gamma} [GeV];Events / bin")
g_data.GetXaxis().SetRangeUser(105, 160)
g_data.SetMinimum(0)

# 背景（蓝色虚线）
f_bkg = TF1("f_bkg", "pol3", 105, 160)
for i in range(4):
    f_bkg.SetParameter(i, f_total.GetParameter(i))
f_bkg.SetLineColor(ROOT.kBlue)
f_bkg.SetLineStyle(2)  # 虚线
f_bkg.Draw("SAME")

# 信号（黑色实线）
f_signal_only = TF1("f_signal_only", "gaus", 105, 160)
f_signal_only.SetParameters(f_total.GetParameter(4), mc_mean, mc_sigma)
f_signal_only.SetLineColor(ROOT.kBlack)  # 黑色信号线
f_signal_only.SetLineStyle(1)  # 实线
f_signal_only.Draw("SAME")

# 总拟合（红色实线）
f_total.SetLineColor(ROOT.kRed)
f_total.SetLineStyle(1)  # 实线
f_total.Draw("SAME")

# 图例
leg = TLegend(0.58, 0.7, 0.88, 0.88)
leg.AddEntry(g_data, "Data (A+B+C+D)", "PE")
leg.AddEntry(f_bkg, "Background", "L")
leg.AddEntry(f_signal_only, "Signal", "L")
leg.AddEntry(f_total, "Signal + Background", "L")
leg.Draw()

# 文本框
latex = TLatex()
latex.SetNDC()
latex.SetTextSize(0.03)
latex.DrawLatex(0.58, 0.65, f"H#rightarrow#gamma#gamma (m_{{H}}={mc_mean:.1f} GeV)")
latex.DrawLatex(0.58, 0.60, f"Signal: {signal_strength:.1f} #pm {signal_error:.1f} events")

c.SaveAs("Higgs_mass.pdf")
input("绘图完成,按回车键退出")