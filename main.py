from trackers.tracker import BallHandler
from utils import read_video, save_video
from trackers import Tracker
from team_assign import TeamAssign
from sports.common.team import TeamClassifier
from split_videos import VideoSplitter
import supervision as sv
import torch
from nicegui import ui, app, background_tasks, run
import webview
import os
from audio_crop import AudioCrop
from contextlib import contextmanager

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'input_video_path': '',
    'output_video_path': '',
    'annotated_video_path': '',
    'player_model_path': '',
    'ball_model_path': '',
    'batch_size': 20,
    'player_to_ball_dist': 70,
    'grace_period': 30,
    'crop_frame_skip': 2,
    'frames_considered_possession': 20,
    'ball_frame_forgiveness': 3,

    'audio_crop': False,
    'ball_track_crop': False,
    'audio_crop_player_name': '',
    'whisper_model': 'small',
    'whisper_processes': 4,
    'audio_start_time_offset': -2,
    'audio_crop_duration': 5,
    'audio_word_similarity': 70,

    'sam_2_mode': False
}

async def choose_file(button, input_video = False,ball_model = False, player_model = False):
    if ball_model or player_model:
        files = await app.native.main_window.create_file_dialog(file_types=["YOLO Model (*.pt)"])
        if files:
            ui.notify("File set")

            if ball_model:
                config['ball_model_path'] = files[0]
                button.classes('bg-green', remove='bg-red')
                print(config['ball_model_path'])

            else:
                config['player_model_path'] = files[0]
                button.classes('bg-green', remove='bg-red')
                print(config['player_model_path'])

    else:
        files = await app.native.main_window.create_file_dialog(file_types=["Video file (*.mp4;*.mov;*.avi;*.mkv;*.flv;*.wmv;*.webm;*.m4v)"])
        if files:
            ui.notify("File set")
            if input_video:
                config['input_video_path'] = files[0]
                button.classes('bg-green', remove='bg-red')
                print(config['input_video_path'])






async def choose_output_folder(button):
    files = await app.native.main_window.create_file_dialog(dialog_type= webview.FOLDER_DIALOG)
    if files:
        ui.notify("Output Folder Set")
        config['output_video_path'] = files[0]
        button.classes('bg-green', remove='bg-red')


def run_program(config):
    # AI detection method via ball tracking
    if not all(config.get(key) not in [None, ''] for key in ['input_video_path', 'output_video_path', 'player_model_path', 'ball_model_path']):
        ui.notify('ERROR: Not all inputs set')
        print(f"RUN PROGRAM ERROR")
        return

    # Output path
    config['annotated_video_path'] = os.path.join(config['output_video_path'], 'output.mp4')

    # Fetch video info
    video_info = sv.VideoInfo.from_video_path(config['input_video_path'])
    w, h = video_info.width, video_info.height

    # Initialize Tracker and TeamAssign
    tracker = Tracker(config['player_model_path'], config['ball_model_path'], w=w, h=h,config= config)
    frame_gen = read_video(config['input_video_path'], config['crop_frame_skip'])
    # team_assigner = TeamAssign(frame_gen, tracker.model)
    #
    # # Extract crops and fit team classifier
    # crops = team_assigner.extract_crops(read_from_stub=True, stub_path='stubs/crop_stub.pk1')
    # team_classifier = TeamClassifier(device=DEVICE)
    # team_classifier.fit(crops)

    # Annotate ball and player positions
    frame_gen = read_video(config['input_video_path'])
    annotated_frames, ball_positions, player_positions = tracker.initialize_and_annotate(
        frame_gen=frame_gen,
        team_classifier=None,
        batch_size=config['batch_size'],
        read_from_stub=False,
        stub_path='stubs/annotation_stub.pk1',
        input_path=config['input_video_path']
    )

    # Handle ball tracking
    ball_handler = BallHandler(
        incomplete_ball_positions=ball_positions,
        annotated_frames=annotated_frames,
        player_positions=player_positions,
        ball_dist=config['player_to_ball_dist']
    )

    ball_annotated_frames, player_in_possession_buffer = ball_handler.handle_ball_tracking(
        read_from_stub=False,
        stub_path='stubs/ball_stub.pk1'
    )

    # Split the video based on possession tracking
    frame_gen = read_video(config['input_video_path'])
    video_splitter = VideoSplitter(
        tracker_array=player_in_possession_buffer,
        frame_gen=frame_gen,
        source_path=config['input_video_path'],
        grace_period=config['grace_period'],
        output_folder=config['output_video_path'],
        frames_considered_possession=config['frames_considered_possession'],
        ball_frame_forgiveness=config['ball_frame_forgiveness']
    )
    video_splitter.crop_videos()

    # Save the final annotated video
    save_video(config['input_video_path'], config['annotated_video_path'], ball_annotated_frames)

    # Mark the process as finished
    print("Done")


async def run_audio_crop_program(config):
    if config['input_video_path'] and config['output_video_path'] and config['audio_crop_player_name']:
        audio_crop = AudioCrop(target_name=config['audio_crop_player_name'],
                               input_file=config['input_video_path'],
                               output_dir=config['output_video_path'],
                               processors=config['whisper_processes'])
        try:
            await run.cpu_bound(audio_crop.start_transcription,
                                config['audio_start_time_offset'],
                                config['audio_crop_duration'],
                                config['audio_word_similarity'])
            ui.notify("Complete")
            print("Done ")
        except Exception as e:
            print('Error during transcription:', e)
            ui.notify(f'Error: {e}')
    else:
        ui.notify('Parameters not filled/correct')



async def run_main_async(button, spinner):
    with disable(button, spinner):
        if config['audio_crop']:
            await run_audio_crop_program(config.copy())
        elif config['ball_track_crop']:
            await run.cpu_bound(run_program, config.copy())
        else:
            ui.notify("Select a method.")

@contextmanager
def disable(button: ui.button, spinner: ui.spinner):
    button.disable()
    spinner.visible = True
    try:
        yield
    finally:
        button.enable()
        spinner.visible = False




def main():
    ui.dark_mode().enable()

    with ui.card():
        with ui.row().classes('w-full justify-center'):
            ui.label("Player Watch:")

        ui.separator()
        # INPUTS
        with ui.grid(columns=2).classes('items-center gap-4'):

            ui.label("Input Video:").classes("self-center")
            input_button = ui.button('Browse', on_click= lambda: choose_file(button= input_button, input_video= True)).classes('text-sm px-6 py-1 bg-red')

            ui.label("Output Folder:").classes("self-center")
            output_button = ui.button('Browse', on_click= lambda: choose_output_folder(output_button)).classes('text-sm px-6 py- bg-red')

            ui.label("Player Detection Model:").classes("self-center")
            player_model_button = ui.button('Browse',on_click= lambda: choose_file(button= player_model_button, player_model= True)).classes('text-sm px-6 py-1 bg-red')

            ui.label("Ball Detection Model:").classes("self-center")
            ball_model_button = ui.button('Browse', on_click= lambda: choose_file(button= ball_model_button, ball_model= True)).classes('text-sm px-6 py-1 bg-red')

        ui.separator()
        # CHOOSE METHOD
        with ui.grid(columns=2).classes('items-center'):
            def toggle_checkbox(self, other_checkbox):
                if self.value:
                    other_checkbox.disable()
                else:
                    other_checkbox.enable()


            ai_audio_crop = ui.checkbox('Audio-Based Clipping', on_change= lambda e: config.update({'audio_crop': e.value}))
            ai_ball_detection = ui.checkbox('Ball-Tracking Clipping', on_change= lambda e: config.update({'ball_track_crop': e.value}))
            ai_audio_crop.on_value_change(lambda: toggle_checkbox(ai_audio_crop, ai_ball_detection))
            ai_ball_detection.on_value_change(lambda: toggle_checkbox(ai_ball_detection, ai_audio_crop))

        # PARAMETER CHOOSING
        # AUDIO METHOD
        ui.separator()
        with ui.row().classes('w-full justify-center'):
            with ui.dropdown_button('Model:', auto_close=True).bind_visibility_from(ai_audio_crop, 'value'):
                ui.item('tiny', on_click=lambda: config.update({'whisper_model': 'tiny'}))
                ui.item('small', on_click=lambda e: config.update({'whisper_model': 'small'}))
                ui.item('medium', on_click=lambda e: config.update({'whisper_model': 'medium'}))
                ui.item('large-v3-turbo', on_click=lambda e: config.update({'whisper_model': 'large-v3-turbo'}))

        with ui.grid(columns=2).classes('items-center').bind_visibility_from(ai_audio_crop, 'value'):
            ui.number(label="Processes",
                      value=config['whisper_processes'],
                      step=1,
                      precision=0,
                      on_change=lambda e: (config.update({'whisper_processes': int(e.value)}),
                                           print(f"Processes set to: {e.value}")))

            ui.number(label="Start time offset",
                      value=config['audio_start_time_offset'],
                      step=1,
                      precision=0,
                      suffix='s',
                      on_change=lambda e: (config.update({'audio_start_time_offset': int(e.value)}),
                                           print(f"Start time offset set to: {e.value}")))

            ui.number(label="Crop length",
                      value=config['audio_crop_duration'],
                      step=1,
                      precision=0,
                      min=0,
                      suffix='s',
                      on_change=lambda e: (config.update({'audio_crop_duration': int(e.value)}),
                                           print(f"Crop duration set to: {e.value}")))
            ui.number(label="Name Match %",
                      value=config['audio_word_similarity'],
                      step=5,
                      precision=0,
                      min=1,
                      max = 100,
                      suffix='%',
                      on_change=lambda e: (config.update({'audio_word_similarity': int(e.value)}),
                                           print(f"Audio word similarity set to: {e.value}")))
        with ui.row().classes('w-full justify-center').bind_visibility_from(ai_audio_crop, 'value'):
            ui.input(placeholder="Shirt Name",
                     validation={"Name too long": lambda value: len(value) < 22},
                     on_change=lambda e: config.update({'audio_crop_player_name': e.value.strip()})).props('rounded outlined dense').props('dense')




        # BALL TRACK METHOD
        with ui.grid(columns=2).classes('items-center gap-4').bind_visibility_from(ai_ball_detection, 'value'):
            ui.number(label="Grace Period",
                      value=config['grace_period'],
                      step=1,
                      precision=0,
                      suffix='fr',
                      on_change=lambda e: (config.update({'grace_period': int(e.value)}),
                                           print(f"Grace period set to: {e.value}")))

            ui.number(label="Player -> Ball distance", 
                      value=config['player_to_ball_dist'],
                      step=1,
                      precision=0,
                      on_change=lambda e: (config.update({'player_to_ball_dist': int(e.value)}),
                                           print(f"Player → Ball distance set to: {e.value}")))

            ui.number(label="Frame Batch Size",
                      value=config['batch_size'],
                      step=5,
                      precision=0,
                      suffix='fr',
                      on_change=lambda e: (config.update({'batch_size': int(e.value)}),
                                           print(f"Batch size set to: {e.value}")))

            ui.number(label="Classifier Frame Skip", value=config['crop_frame_skip'],
                      step=1,
                      precision=0,
                      suffix='fr',
                      on_change=lambda e: (config.update({'crop_frame_skip': int(e.value)}),
                                           print(f"Classifier frame skip set to: {e.value}")))

            ui.number(label="Possession threshold",
                      value=config['frames_considered_possession'],
                      step=1,
                      precision=0,
                      suffix='fr',
                      on_change=lambda e: (config.update({'frames_considered_possession': int(e.value)}),
                                           print(f"Possession threshold set to: {e.value}")))

            ui.number(label="Ball Error Forgiveness",
                      value=config['ball_frame_forgiveness'],
                      step=1,
                      precision=0,
                      suffix='fr',
                      on_change=lambda e: (config.update({'ball_frame_forgiveness': int(e.value)}),
                                           print(f"Ball frame forgiveness set to: {e.value}")))
        with ui.row().classes('w-full justify-center').bind_visibility_from(ai_ball_detection, 'value'):
            sam_2_mode = ui.checkbox('SAM-2 Ball Track',
                                        on_change=lambda e: config.update({'sam_2_mode': e.value}))

        ui.separator()
        with ui.row().classes('w-full justify-center'):
            run_button = ui.button('Run', on_click=lambda e: run_main_async(run_button, processing_spinner))

        with ui.row().classes('w-full justify-center'):
            processing_spinner = ui.spinner()
            processing_spinner.visible = False




    app.native.window_args['resizable'] = False
    ui.run(native=True, window_size=(430,830), reload= False)



if __name__ in "__main__":
    main()