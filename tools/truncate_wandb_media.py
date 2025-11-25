import wandb

def truncate_wandb_projects(entity, project_list):
    api = wandb.Api()
    for project in project_list:
        print(f"Processing project: {project}")
        try:
            runs = api.runs(f"{entity}/{project}")
            for run in runs:
                print(f"  Processing run: {run.id}")
                for file in run.files():
                    if file.name.startswith("media/") and (file.name.endswith(".png") or file.name.endswith(".jpg")):
                        print("    Deleting:", file.name)
                        file.delete()
                print("  Media cleanup finished for", run.id)
        except Exception as e:
            print(f"Error processing project {project}: {e}")

if __name__ == "__main__":
    # 專案名稱列表
    project_list = [
        "74_cityscapesA010_fedprox_4C_12R_202511110010",
        "68_kittiOA010_fednova_10C_12R_202511042259",
        "70_kittiOA010_fedyoga_10C_12R_202511042301",
        "67_kittiOA010_fedprox_10C_12R_202511042258",
        "47_kittiOA010_fedavg_10C_12R_202511040805",
        "13_kitti_4C_4R_202508161102",
        "14_kitti_4C_4R_202508161656",
        "4_kitti_5C_3R_202508071708",
        "23_cocoA010_4C_6R_202510151813",
        "23_sim10k_4C_3R_202509031403",
        "82_cityscapes_fedyoga_4C_12R_202511110019",
        "21_cityscapes_4C_3R_202509031309",
        "22_foggy_4C_3R_202509031311",
        "23_kittiO_fednova_4C_5R_202510290346",
        "21_kittiO_fedavg_4C_5R_202510282248",
        "31_kittiOA010_fedawa_4C_5R_202510300347",
        "28_kittiOA010_fednova_4C_5R_202510292015",
        "32_kittiOA010_fedyoga_4C_5R_202510300613",
        "78_cityscapes_fedavgm_4C_12R_202511110015",
        "30_kittiOA010_fedprox_4C_5R_202510300121",
        "27_kittiOA010_fedavgm_4C_5R_202510291750",
        "26_kittiOA010_fedavg_4C_5R_202510291305",
        "19_kitti_4C_3R_202508292229",
        "18_kitti_4C_3R_202508292227",
        "17_kitti_4C_3R_202508292132"
    ]

    truncate_wandb_projects("nchc", project_list)
