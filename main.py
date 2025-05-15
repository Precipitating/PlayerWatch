import asyncio
from trackers.tracker import BallHandler
from utils import read_video
from trackers import Tracker
from split_videos import VideoSplitter
import supervision as sv
import torch
from nicegui import ui, app, background_tasks, Client
import webview
import os
from audio_crop import AudioCrop
from contextlib import contextmanager
import glob
from concurrent.futures import ProcessPoolExecutor
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'input_video_path': '',
    'output_video_path': '',
    'annotated_video_path': '',
    'annotated_players_path': 'stubs/annotated_players.mp4',
    'player_model_path': '',
    'ball_model_path': '',
    'batch_size': 4,
    'player_to_ball_dist': 70,
    'grace_period': 30,
    'save_output_video': False,
    'frames_considered_possession': 20,
    'ball_frame_forgiveness': 3,
    'slice_threads': 1,

    'audio_crop': False,
    'ball_track_crop': False,
    'audio_crop_player_name': '',
    'whisper_model': 'small',
    'whisper_batch_size': 4,
    'audio_start_time_offset': -2,
    'audio_crop_duration': 5,
    'audio_word_similarity': 70,

    'sam_2_mode': False,
    'sam_2_model_path': ''
}
"""
Opens a windows file browser asking for a specified file format
Args:
    button (ui.button): The button linked to this function. Used for turning its colour green if file is selected.
    input_video (bool): Is input video button? (*mp4/avi etc file)
    ball_model (bool): Is ball model button? (*PT file)
    player_model (bool): Is player model button?  (*PT file)
    sam_2_model (bool): Is SAM2 model button?  (*PT file)
    
    
Saves:
    The path of the file to the linked config key
"""


async def choose_file(button, input_video=False, ball_model=False, player_model=False, sam_2_model=False):
    if ball_model or player_model or sam_2_model:
        model_type = 'ball_model' if ball_model else 'player_model' if player_model else 'sam_2_model'
        model_path = await app.native.main_window.create_file_dialog(file_types=["PT Model (*.pt)"])

        if model_path:
            config[f'{model_type}_path'] = model_path[0]
            button.classes('bg-green', remove='bg-red')
            print(config[f'{model_type}_path'])
            ui.notify("File set")
    else:
        video_path = await app.native.main_window.create_file_dialog(
            file_types=["Video file (*.mp4;*.mov;*.avi;*.mkv;*.flv;*.wmv;*.webm;*.m4v)"])

        if video_path:
            config['input_video_path'] = video_path[0]
            button.classes('bg-green', remove='bg-red')
            print(config['input_video_path'])
            ui.notify("File set")


"""
Opens a windows file browser asking a folder
Args:
    button (ui.button): The button linked to this function. Used for turning its colour green if file is selected.
Saves:
    The path of the file to the config['output_video_path']
"""


async def choose_output_folder(button):
    files = await app.native.main_window.create_file_dialog(dialog_type= webview.FOLDER_DIALOG)
    if files:
        ui.notify("Output Folder Set")
        config['output_video_path'] = files[0]
        button.classes('bg-green', remove='bg-red')


"""
Deletes every pkl (pickle) file or mp4 file in the stub directory if it exists.
"""
def delete_working_files():
    stub_path = os.path.join(os.getcwd(), 'stubs')
    if os.path.isdir(stub_path):
        for ext in ['*.pkl', '*.mp4']:
            files = glob.glob(os.path.join(stub_path, ext))
            for file in files:
                os.remove(file)
                print(f"Deleted: {file}")


"""
Checks for errors before the ball tracking method runs 

Returns:
    True if everything is set correctly
    False if a requirement isn't correct
"""
def ball_tracking_error_checking():
    # ERROR CHECKING

    # Ball model or SAM 2 model needs to be set
    if all(config.get(key) in [None, ''] for key in ['ball_model_path', 'sam_2_model_path']):
        notify('ERROR: Set ball model or SAM 2 model')
        print(f"Ball model or SAM model not set")
        return False

    # SAM 2 needs to be set if SAM 2 option is checked
    if config['sam_2_mode']:
        if not config['sam_2_model_path']:
            notify('ERROR: SAM2 model is required')
            print(f"SAM Model not set")
            return False

    # Checks generic requirements: Input video, output folder and player ball detection model
    if any(config.get(key) in [None, ''] for key in ['input_video_path', 'output_video_path', 'player_model_path']):
        notify('ERROR: Not all inputs set')
        print(f"RUN PROGRAM ERROR")
        return False

    return True


"""
An extension to the ui.notify that works on background threads.
This sends the ui.notify to every client, so ui.notify works even if inside a different process.
"""
def notify(msg: str):
    for client in Client.instances.values():
        with client:
            ui.notify(msg)
"""
The main ball tracking method function
Steps:
    1. Stores a generator composed of all the frames of input_video_path in frame_gen.
    
    2. Initializes tracker & calls process_player_and_ball_tracking which handles player and ball tracking
       (creates an mp4 file of annotated players if save_output_video, and pickle files for later use)
       
    3. Initializes BallHandler responsible for filling missing ball tracking data, getting the player in possession per frame and
       ball annotation (if save_output_video) via handle_ball_tracking function.

    4. Initializes VideoSplitter class, relying on player_in_possession_buffer.pkl to crop the video, depending if a player
       is possessing the ball for config['frames_considered_possession'] frames (start).
       The crop end is finding the next possessor possessing for the same amount of frames.
       
Args:
    config (dict): A copy of config passed through as this function will be run in a different process.
"""
def run_program(config):
    # delete previous working files if applicable
    delete_working_files()

    # Get video data
    video_info = sv.VideoInfo.from_video_path(config['input_video_path'])
    width, height = video_info.width, video_info.height

    # Output path
    config['annotated_video_path'] = os.path.join(config['output_video_path'], 'output.mp4')

    # Initialize Tracker
    tracker = Tracker(config['player_model_path'], config['ball_model_path'], w=width, h=height,config= config)

    # Annotate ball and player positions, and get the player in possession per frame buffer
    frame_gen = read_video(config['input_video_path'])
    tracker.process_player_and_ball_tracking(frame_gen=frame_gen)

    print("Initialize and annotate DONE")

    # Handle ball tracking
    ball_handler = BallHandler(
        ball_dist=config['player_to_ball_dist'],
        config= config
    )

    ball_handler.handle_ball_tracking()
    print("Ball tracking DONE")
    # Split the video based on possession tracking
    frame_gen = read_video(config['input_video_path'])
    video_splitter = VideoSplitter(
        frame_gen=frame_gen,
        source_path=config['input_video_path'],
        grace_period=config['grace_period'],
        output_folder=config['output_video_path'],
        frames_considered_possession=config['frames_considered_possession'],
        ball_frame_forgiveness=config['ball_frame_forgiveness']
    )
    video_splitter.crop_videos()

    # Mark the process as finished
    delete_working_files()

    notify("Complete")
    print("Done")


"""
The main audio tracking method function
Steps:
    1. Checks if all the relevant inputs have been selected.

    2. Initializes AudioCrop responsible for handling the audio crop process, using faster_whisper
       to find config['audio_crop_player_name'], which marks the start (+ config['audio_start_time_offset']).
       It then uses a fixed duration config['audio_crop_duration'] to mark the end (no reliable method to find the next possessor)

Args:
    config (dict): A copy of config passed through as this function will be run in a different process.
"""
async def run_audio_crop_program(config):
    loop = asyncio.get_running_loop()
    if all(config.get(k) for k in ['input_video_path', 'output_video_path', 'audio_crop_player_name']):
        audio_crop = AudioCrop(target_name=config['audio_crop_player_name'],
                               input_file=config['input_video_path'],
                               output_dir=config['output_video_path'],
                               batch_size=config['whisper_batch_size'],
                               model_size=config['whisper_model'])
        try:
            with ProcessPoolExecutor() as executor:
                await loop.run_in_executor(executor, audio_crop.start_transcription,
                                    config['audio_start_time_offset'],
                                    config['audio_crop_duration'],
                                    config['audio_word_similarity'])
            notify("Complete")
            print("Done ")
        except Exception as e:
            print('Error during transcription:', e)
            notify(f'Error: {e}')
    else:
        notify('ERROR: Parameters not filled/correct')


"""
The function that gets called when the RUN button is clicked
This function determines which method function to run, and disables/re-enables the button when necessary (via disable function)

Args:
    button (ui.button): The run button, used for disabling and re-enabling the button when a method is finished.
    spinner (ui.spinner): A spinner widget that shows up when a method is running, disabled when finished.
"""
async def run_main_async(button, spinner):
    loop = asyncio.get_running_loop()
    with disable(button, spinner):
        if config['audio_crop']:
            await run_audio_crop_program(config.copy())
        elif config['ball_track_crop']:
            if ball_tracking_error_checking():
                try:
                    with ProcessPoolExecutor() as executor:
                        await loop.run_in_executor(executor,run_program, config.copy())

                    #await run.cpu_bound(run_program, config.copy())
                except Exception as e:
                    print('Error:', e)
                    notify(f'ERROR: {e}')
        else:
            notify("ERROR: Select a method.")

"""
Function responsible for disabling/re-enabling the spinner and button using a context manager (allows the use of with:)
Args:
    button (ui.button): The run button, Enabled on setup and disabled on cleanup
    spinner (ui.spinner): Visible on setup and disabled on cleanup
"""
@contextmanager
def disable(button: ui.button, spinner: ui.spinner):
    button.disable()
    spinner.visible = True
    try:
        yield
    finally:
        button.enable()
        spinner.visible = False



"""
The main UI code, using NiceGUI as the front end.
"""
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

            ui.label("SAM 2 Model:").classes("self-center")
            sam_2_model = ui.button('Browse', on_click= lambda: choose_file(button= sam_2_model, sam_2_model= True)).classes('text-sm px-6 py-1 bg-red')

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
                ui.item('base', on_click=lambda e: config.update({'whisper_model': 'base'}))
                ui.item('large-v3', on_click=lambda e: config.update({'whisper_model': 'large-v3'}))

        with ui.grid(columns=2).classes('items-center').bind_visibility_from(ai_audio_crop, 'value'):
            ui.number(label="Batch Size",
                      value=config['whisper_batch_size'],
                      step=1,
                      precision=0,
                      on_change=lambda e: (config.update({'whisper_batch_size': int(e.value)}),
                                           print(f"Batch size set to: {e.value}")))

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
            ui.number(label="Slice Threads",
                      value=config['slice_threads'],
                      step=1,
                      precision=0,
                      on_change=lambda e: (config.update({'slice_threads': int(e.value)}),
                                           print(f"Slice Threads set to: {e.value}")))
        with ui.grid(columns=2).classes('items-center gap-4').bind_visibility_from(ai_ball_detection, 'value'):

            ui.checkbox('SAM-2 Ball Track',
                                        on_change=lambda e: config.update({'sam_2_mode': e.value}))
            ui.checkbox(text="Save debug video",
                        value=config['save_output_video'],
                        on_change=lambda e: (config.update({'save_output_video': e.value}),
                                           print(f"Save annotated video set to: {e.value}")))

        ui.separator()
        with ui.row().classes('w-full justify-center'):
            run_button = ui.button('Run', on_click=lambda e: background_tasks.create(run_main_async(run_button, processing_spinner)))

        with ui.row().classes('w-full justify-center'):
            processing_spinner = ui.spinner()
            processing_spinner.visible = False




    app.native.window_args['resizable'] = False
    ui.run(native=True, window_size=(430,830), reload= False, title="PlayerWatch")



if __name__ in "__main__":
    main()