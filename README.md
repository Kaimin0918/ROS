# 基於深度強化學習之ROS自主巡邏與Web遠端控制系統
本專案實作以 SAC 為核心的自主巡邏導航，並提供瀏覽器端 Web 控制與狀態視覺化，透過 rosbridge 與 web_video_server 實現即時指令與影像串流；提供模擬到實機的 Sim‑to‑Real 流程與最小可重現範例。
主要功能
SAC 導航策略：訓練與推論節點，支援地圖定位與避障，提供模型切換與參數化設定。
Web 遙控與監看：使用 roslib.js 連線 ws://<ROS_IP>:9090，按鍵/按鈕發布 /patrol_web_cmd，訂閱 /patrol_web_status 與影像流。
一鍵啟動：simulation_patrol.launch 同時啟動地圖、AMCL/導航、rosbridge、web_video_server 與巡邏管理節點。
Sim‑to‑Real：提供從 Gazebo 到 TurtleBot3 的轉移步驟與調參建議。

## 如何重現
- 安裝 ROS Noetic、Gazebo 與 rosbridge_suite、web_video_server；放置模型/地圖至指定資料夾。
- roslaunch simulation_patrol.launch 啟動系統；確認 ws://<ROS_IP>:9090 及影像端點可用。
- 以 python3 -m http.server 8000 開啟 web/，用瀏覽器載入 index.html 並連線 WebSocket，測試遙控與巡邏模式。

## 相關主題
指令輸入：/patrol_web_cmd、/cmd_vel；狀態輸出：/patrol_web_status；導航/地圖：/map、/tf、/move_base_simple/goal。
待辦與問題
前端 UI 優化與本地庫管理（roslib.js、ros2d.js、easeljs、eventemitter2）。
