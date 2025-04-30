import cv2

from trackers.tracker import BallTracker, BallHandler
from utils import read_video, save_video
from trackers import Tracker
from team_assign import TeamAssign
from sports.common.team import TeamClassifier
from split_videos import VideoSplitter
import supervision as sv
import torch
from nicegui import ui, app, background_tasks
import webview

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

input_video_path = ''
output_video_path = ''
annotated_video_path = ''
player_model_path = ''
ball_model_path = ''
batch_size = 20
player_to_ball_dist = 70
grace_period = 30
crop_frame_skip = 2
frames_considered_possession = 20
ball_frame_forgiveness = 3
is_running = False


def set_batch_size(val):
    global batch_size
    batch_size = int(val)
    print(f"Batch size set to: {val}")

def set_player_ball_dist(val):
    global player_to_ball_dist
    player_to_ball_dist = int(val)
    print(f"Player ball dist set to: {val}")

def set_grace_period(val):
    global batch_size
    batch_size = int(val)
    print(f"Grace period set to: {val}")

def set_crop_frame_skip(val):
    global crop_frame_skip
    crop_frame_skip = int(val)
    print(f"Crop frame skip set to: {val}")

def set_frames_considered_possession(val):
    global frames_considered_possession
    frames_considered_possession = int(val)
    print(f"Frames considered possession set to: {val}")

def set_ball_frame_forgiveness(val):
    global ball_frame_forgiveness
    ball_frame_forgiveness = int(val)
    print(f"Ball frame forgiveness set to: {val}")

def set_input_video_path(path):
    global input_video_path
    input_video_path = path
    print(f"Input video path set to: {input_video_path}")


def set_output_video_path(path):
    global output_video_path
    output_video_path = path
    print(f"Output video path set to: {output_video_path}")

def set_ball_model_path(path):
    global ball_model_path
    ball_model_path = path
    print(f"Ball model path set to: {ball_model_path}")



def set_player_model_path(path):
    global player_model_path
    player_model_path = path
    print(f"Player model path set to: {player_model_path}")

async def choose_file(input_video = False,ball_model = False, player_model = False):
    if ball_model or player_model:
        files = await app.native.main_window.create_file_dialog(file_types=["YOLO Model (*.pt)"])
        if files:
            ui.notify("File set")

            if ball_model:
                set_ball_model_path(files[0])
            else:
                set_player_model_path(files[0])

        else:
            ui.notify("File not set")
    else:
        files = await app.native.main_window.create_file_dialog(file_types=["Video file (*.mp4;*.mov;*.avi;*.mkv;*.flv;*.wmv;*.webm;*.m4v)"])
        if files:
            ui.notify("File set")
            if input_video:
                set_input_video_path(files[0])
        else:
            ui.notify("File not set")






async def choose_output_folder():
    files = await app.native.main_window.create_file_dialog(dialog_type= webview.FOLDER_DIALOG)
    if files:
        ui.notify("Output Folder Set")
        set_output_video_path(files[0])
    else:
        ui.notify("Output Folder Not Set")




def run_program():
    global is_running, annotated_video_path
    if not is_running and all([input_video_path,
                               output_video_path,
                               player_model_path,
                               ball_model_path]):
        is_running = True
        annotated_video_path = f"{output_video_path}\\output.mp4"
        video_info = sv.VideoInfo.from_video_path(input_video_path)
        w, h = video_info.width, video_info.height
        tracker = Tracker(player_model_path,ball_model_path, w= w, h= h)

        # assign team colors
        frame_gen= read_video(input_video_path, crop_frame_skip)

        team_assigner = TeamAssign(frame_gen, tracker.model)
        crops = team_assigner.extract_crops(read_from_stub= True, stub_path='stubs/crop_stub.pk1')
        team_classifier = TeamClassifier(device=DEVICE)
        team_classifier.fit(crops)

        # read video
        frame_gen = read_video(input_video_path)

        # initialize model and annotate ball and player
        annotated_frames, ball_positions, player_positions = tracker.initialize_and_annotate(frame_gen= frame_gen,
                                                                                             team_classifier= team_classifier,
                                                                                             batch_size= batch_size,
                                                                                             read_from_stub= True,
                                                                                             stub_path= 'stubs/annotation_stub.pk1')



        ball_handler = BallHandler(incomplete_ball_positions= ball_positions,
                                   annotated_frames= annotated_frames,
                                   player_positions= player_positions,
                                   ball_dist= player_to_ball_dist)

        ball_annotated_frames, player_in_possession_buffer = ball_handler.handle_ball_tracking(read_from_stub= True,
                                                                                               stub_path = 'stubs/ball_stub.pk1')


        print(player_in_possession_buffer)
        frame_gen = read_video(input_video_path)

        video_splitter = VideoSplitter(tracker_array= player_in_possession_buffer,
                                       frame_gen= frame_gen,
                                       source_path= input_video_path,
                                       grace_period= grace_period,
                                       output_folder= output_video_path,
                                       frames_considered_possession= frames_considered_possession,
                                       ball_frame_forgiveness= ball_frame_forgiveness
                                       )
        video_splitter.crop_videos()

        save_video(input_video_path, annotated_video_path, ball_annotated_frames)
        is_running = False
        print("Done")
    else:
        print("RUN PROGRAM ERROR")




def main():
    ui.dark_mode().enable()
    with ui.card():
        with ui.row().classes('w-full justify-center'):
            ui.label("Player Watch:")

        ui.separator()

        with ui.grid(columns=2).classes('items-center gap-4'):

            ui.label("Input Video:").classes("self-center")
            ui.button('Browse', on_click= lambda: choose_file(input_video= True)).classes('text-sm px-6 py-1')

            ui.label("Output Folder:").classes("self-center")
            ui.button('Browse', on_click= choose_output_folder).classes('text-sm px-6 py-1')

            ui.label("Player Detection Model:").classes("self-center")
            ui.button('Browse',on_click= lambda: choose_file(player_model= True)).classes('text-sm px-6 py-1')

            ui.label("Ball Detection Model:").classes("self-center")
            ui.button('Browse', on_click= lambda: choose_file(ball_model= True)).classes('text-sm px-6 py-1')


        ui.separator()

        with ui.grid(columns=2).classes('items-center gap-4'):
            ui.number(label="Grace Period (frames)", value= grace_period, step=1, precision=0, on_change= lambda e: set_grace_period(e.value))
            ui.number(label="Player -> Ball distance", value= player_to_ball_dist, step=1, precision=0, on_change= lambda e: set_player_ball_dist(e.value) )
            ui.number(label="Frame Batch Size", value= batch_size, step=5, precision=0, on_change= lambda e: set_batch_size(e.value))
            ui.number(label="Classifier Frame Skip", value=crop_frame_skip, step=1, precision=0, on_change= lambda e: set_crop_frame_skip(e.value))
            ui.number(label="Possession threshold", value= frames_considered_possession, step=1, precision=0, on_change= lambda e: set_frames_considered_possession(e.value))
            ui.number(label="Ball Error Forgiveness", value=ball_frame_forgiveness, step=1, precision=0, on_change= lambda e: set_ball_frame_forgiveness(e.value))

        ui.separator()
        with ui.row().classes('w-full justify-center'):
            ui.button('Run', on_click= run_program)



    app.native.window_args['resizable'] = False
    ui.run(native=True, window_size=(430,790))






    # input_video_path = 'input_videos/sample.mp4'
    # output_video_path = 'output_videos/output.mp4'
    # video_info = sv.VideoInfo.from_video_path(input_video_path)
    # w, h = video_info.width, video_info.height
    # tracker = Tracker('models/main/repobest.pt','models/ball/640/best.pt', w= w, h= h)
    #
    #
    # # assign team colors
    # frame_gen= read_video(input_video_path, 2)
    #
    # team_assigner = TeamAssign(frame_gen, tracker.model)
    # crops = team_assigner.extract_crops(read_from_stub= True, stub_path='stubs/crop_stub.pk1')
    # team_classifier = TeamClassifier(device=DEVICE)
    # team_classifier.fit(crops)
    #
    # # read video
    # frame_gen = read_video(input_video_path)
    #
    # # initialize model and annotate ball and player
    # annotated_frames, ball_positions, player_positions = tracker.initialize_and_annotate(frame_gen= frame_gen,
    #                                                                                      team_classifier= team_classifier,
    #                                                                                      batch_size= 20,
    #                                                                                      read_from_stub= True,
    #                                                                                      stub_path= 'stubs/annotation_stub.pk1')
    #
    #
    #
    # ball_handler = BallHandler(incomplete_ball_positions= ball_positions,
    #                            annotated_frames= annotated_frames,
    #                            player_positions= player_positions)
    #
    # ball_annotated_frames, player_in_possession_buffer = ball_handler.handle_ball_tracking(read_from_stub= True, stub_path = 'stubs/ball_stub.pk1')
    #
    #
    #
    #
    # print(player_in_possession_buffer)
    # frame_gen = read_video(input_video_path)
    #
    # video_splitter = VideoSplitter(tracker_array= player_in_possession_buffer, frame_gen= frame_gen, source_path= input_video_path)
    # video_splitter.crop_videos()
    #
    #
    # save_video(input_video_path, output_video_path, ball_annotated_frames)





if __name__ in {"__main__", "__mp_main__"}:
    main()