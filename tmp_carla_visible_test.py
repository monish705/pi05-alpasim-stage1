import queue, time
import carla, numpy as np, pygame
WIDTH, HEIGHT = 1280, 720
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(20.0)
world = None
for i in range(24):
    try:
        world = client.get_world()
        print(f'connected on try {i+1}: {world.get_map().name}', flush=True)
        break
    except Exception as e:
        print(f'connect try {i+1} failed: {e}', flush=True)
        time.sleep(5)
if world is None:
    raise RuntimeError('CARLA never became ready')
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('CARLA Visible Test')
clock = pygame.time.Clock()
q = queue.Queue()
bp = world.get_blueprint_library().filter('model3')[0]
vehicle = None
for sp in world.get_map().get_spawn_points()[:20]:
    vehicle = world.try_spawn_actor(bp, sp)
    if vehicle is not None:
        break
if vehicle is None:
    raise RuntimeError('Could not spawn model3')
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '100')
camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=-6.0, z=2.5), carla.Rotation(pitch=-10.0)), attach_to=vehicle)
vehicle.set_autopilot(True)
def on_image(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    q.put(arr)
camera.listen(on_image)
last = None
start = time.time()
try:
    while time.time() - start < 300:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
        try:
            last = q.get(timeout=2.0)
        except queue.Empty:
            continue
        screen.blit(pygame.surfarray.make_surface(last.swapaxes(0,1)), (0,0))
        pygame.display.flip()
        clock.tick(20)
finally:
    if last is not None:
        pygame.image.save(screen, '/home/ubuntu/carla_visible_test.png')
    camera.stop(); camera.destroy(); vehicle.destroy(); pygame.quit()
