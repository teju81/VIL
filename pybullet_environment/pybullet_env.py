import pybullet as p
import time
import pybullet_data
import gym
from gym import spaces
from gym.utils import seeding
import os
import os.path as path
import numpy as np
from gym.envs.pybullet.pybullet_camera import PyBCamera
from PIL import Image, ImageDraw, ImageFont
import random
import glob
import numpy as np
from sklearn.cluster import KMeans
import imageio
import matplotlib.pyplot as plt
RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class PybulletMujocoXmlEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """
    metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': 60
        }
    def __init__(self, model_path, robot_name, action_dim, obs_dim, renders=True):
        self._timeStep = 1./60.
        self._fps = 60
        self.isRender = renders
        self.self_collision = True
        self.terminated = 0

        
        self.stateId = -1
        self.tableID = None
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.useRealTimeSimulation = 1

        self.render_width = RENDER_WIDTH
        self.render_height = RENDER_HEIGHT
        self.gif_width = 256
        self.gif_height = 256
        self.T = 24
        self._p = None
        self.model_path = model_path
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.parts = None
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None
        self.robot_name = robot_name

        # Defining cameras
        self._cam_target = [0, 0, 0]
        self._upVector = [0.0, 0.0, 0.7]
        self._cam_dist = 1.5
        self._cam_yaw = 90
        self._cam_pitch = -89
        self._cam_pos = [0.0, 0.0, 0.0]
        self._fov = 30
        self._near = 0.1
        self._far = -0.1
        self._aspect = 1
        self._cam_roll = 0
        self._cam_upaxisindex = 2
        
        self.training_phase = True
        self.dataset_type = "colors" # Other option is omniglot
        self.object_type = "square"
        self.num_objects = 4
        self.target_color = (0,0,255)
        self.object_scale = 20
        self.color_pallete = [(c1,c2,c3) for c1 in range(0,255,20) for c2 in range(0,255,20) for c3 in range(0,255,20)]
        self.color_list = None
        self.scale_list = [20]*self.num_objects

        # initialising environment
        self.seed()

        # self.reset_env()
        # self.state = self.reset()

        # self.cam = [None] * 1
        # self.cam[0] = PyBCamera(threadID = 0, p = self._p, cameraPos = self._cam_pos, target = self._camera_target, upVector = self._upVector, fov = self._fov, aspect = self._aspect, near = self._near, far = self._far)

        # Define Action space and observation space
        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf*np.ones([self.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        self.viewer = None

    def close_2(self):
        if self._p is not None:
            self._p.disconnect()
            self._p = None

    def reset_env(self, initial_state = None):
        self.close_2()
        self._p = p
        if self.isRender:
            physicsClient = self._p.connect(self._p.GUI, options='--width=%d --height=%d' % (self.render_width, self.render_height))
        else:
            physicsClient = self._p.connect(self._p.DIRECT, options='--width=%d --height=%d' % (self.render_width, self.render_height))
            
        self._p.setRealTimeSimulation(self.useRealTimeSimulation)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.terminated = 0
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        self._p.setTimeStep(self._timeStep)
        
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING,1)
        self._p.resetDebugVisualizerCamera(cameraDistance=self._cam_dist, cameraYaw=self._cam_yaw, cameraPitch=self._cam_pitch, cameraTargetPosition=self._cam_target)
        self._p.setGravity(0,0,-9.81)    
        self.viewMatrix = self._p.computeViewMatrixFromYawPitchRoll(self._cam_pos, distance=self._cam_dist, yaw=self._cam_yaw, pitch=self._cam_pitch, roll=self._cam_roll, upAxisIndex=self._cam_upaxisindex)
        self.projMatrix = self._p.computeProjectionMatrixFOV(self._fov, self._aspect, self._near, self._far)

        #Robot Load
        if self.model_path.startswith("/"):
            fullpath = self.model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), self.model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        if self.self_collision:
            self.model = self._p.loadMJCF(fullpath, flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        else:
            self.model = self._p.loadMJCF(fullpath)
        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self.model)

        # kuka = p.loadURDF("kuka_iiwa/model_vr_limits.urdf", basePosition=(0.375, -0.6, -0.1), baseOrientation=(0.0, 0.0, 0.0, 1.0))
        # jointPositions=[0.0, 0.0, 0.0, 1.57, 0.0, -1.036, 0.0]
        # for jointIndex in range (p.getNumJoints(kuka)):
        #     p.resetJointState(kuka, jointIndex, jointPositions[jointIndex])
        #     p.setJointMotorControl2(kuka, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex], 0)

        # kuka_gripper = p.loadSDF("gripper/wsg50_one_motor_gripper_new_free_base.sdf")[0]
        # p.resetBasePositionAndOrientation(kuka_gripper, [0.923103, -0.2, 1.250036],[0, 0.964531, 0, -0.263970])
        # jointPositions=[0, -0.011130, -0.206421, 0.205143, -0.01, 0, -0.01, 0]
        # for jointIndex in range (p.getNumJoints(kuka_gripper)):
        #     p.resetJointState(kuka_gripper, jointIndex, jointPositions[jointIndex])
        #     p.setJointMotorControl2(kuka_gripper, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex], 0)

        # kuka_cid = p.createConstraint(kuka, 6, kuka_gripper, 0, p.JOINT_FIXED, [0,0,0], [0,0,0.05], [0,0,0])

        #Reset Base position and orientation (State)
        if self.dataset_type == "colors":
            self.spawn_colors_on_table(num_objects = self.num_objects, object_type=self.object_type, color_mode = 2, scale_mode = 2, random_table = True)
        else:
            self.spawn_images_on_table()
        self.robot_specific_reset(initial_state)

    def DrawObject(self, draw_obj, object_type = "circle", color = (0,0,255), pos = (512,384), scale = 10):
        px = pos[0]
        py = pos[1]
        if object_type == "circle":
            x0 = px - scale
            y0 = py - scale
            x1 = px + scale
            y1 = py + scale
            draw_obj.ellipse([(x0,y0), (x1, y1)], fill = color)
        elif object_type == "cross":
            x0 = px - scale
            y0 = py - scale
            x1 = px + scale
            y1 = py + scale
            draw_obj.line([(x0,y0), (x1, y1)], fill = color, width = 10)
            draw_obj.line([(x0,y1), (x1, y0)], fill = color, width = 10)
        elif object_type == "square":
            x0 = px - scale
            y0 = py - scale
            x1 = px + scale
            y1 = py + scale
            draw_obj.rectangle([(x0,y0), (x1, y1)], fill = color)
        elif object_type == "triangle":
            x0 = px - scale
            y0 = py
            x1 = px
            y1 = py + 1.732*scale
            x2 = px + scale
            y2 = py
            draw_obj.polygon([(x0,y0), (x1, y1), (x2,y2)], fill = color)
        elif object_type == "star":
            x0 = px - scale
            y0 = py
            x1 = px
            y1 = py + 1.732*scale
            x2 = px + scale
            y2 = py
            draw_obj.polygon([(x0,y0), (x1, y1), (x2,y2)], fill = color)
            x0 = px - scale
            y0 = py + 0.65*1.732*scale
            x1 = px
            y1 = py - 0.35*1.732*scale
            x2 = px + scale
            y2 = py + 0.65*1.732*scale
            draw_obj.polygon([(x0,y0), (x1, y1), (x2,y2)], fill = color)

    # def spawn_objects_on_table(self, num_objects = 3):
    #     image = Image.open("table/table_original.png")
    #     draw = ImageDraw.Draw(image)

    #     # Randomly choose the objects and their colors
    #     object_list = ["circle", "cross", "square", "triangle", "star"]
    #     object_types = random.sample(object_list,num_objects)

    #     color_list = {"red":(255,0,0,255), "green":(0,255,0,255), "blue":(0,0,255,255)}
    #     colors = random.sample(list(color_list.values()),num_objects)

    #     for i in range(num_objects):
    #         self.find_random_point()
    #         if i == 0:
    #             self.target_position = [self.posx, self.posy, 0.01] # Setting target to a point close to the object, not necessarily its centre
    #         # Draw 2D shape on Table texture image
    #         self.coord2pix()
    #         self.DrawObject(draw_obj=draw, object_type=object_types[i], color=colors[i])

    #     # Save the drawn patterns to texture that will be loaded
    #     image.save("table/table.png")

    #     # Load Table
    #     self.tableID = self._p.loadURDF("table/table.urdf", basePosition=[0, 0, -0.61], baseOrientation=[0, 0, 0, 1])

    def DrawText(self, draw_obj, text = "Z", color=(0,0,255,255)):
            fnt = ImageFont.truetype("Karenni Font 4.ttf", 15)
            #fnt = ImageFont.load_default()
            draw_obj.text((self.px,self.py), text, font=fnt, fill = color)

    def DrawImage(self, img_obj, img_path):
            labels = self.segment_image(img_path)
            w = 100
            h = 100
            img = Image.open(img_path)
            img = img.resize((w,h))
            x = int(self.px-w/2)
            y = int(self.py-h/2)
            img_obj.paste(img, (x,y))

    def is_overlap(self, p1, p2, w, h):
        # P1 and P2 are tuples
        # First Square
        #Bottom Left Corner
        l1x = p1[0] - int(w/2)
        l1y = p1[1] - int(h/2)
        # Top Right Corner
        r1x = p1[0] + int(w/2)
        r1y = p1[1] + int(h/2)

        # Second Square
        #Bottom Left Corner
        l2x = p2[0] - int(w/2)
        l2y = p2[1] - int(h/2)
        # Top Right Corner
        r2x = p2[0] + int(w/2)
        r2y = p2[1] + int(h/2)

        # No Overlap: If one of the rectangles is to the left of the other
        if (l2x > r1x) or (l1x > r2x):
            return False
        # No Overlap: If one of the rectangles is above the other
        if (r1y < l2y) or (r2y < l1y):
            return False

        return True # Overlap

    def find_random_point(self):
        self.posx = self.np_random.uniform(low=-0.27, high=0.27)
        self.posy = self.np_random.uniform(low=-0.27, high=0.27)
        return self.posx, self.posy

    def coord2pix(self):
        coord2pix_x = 512/0.375
        coord2pix_y = 384/0.25
        self.px = int(512 + 0.5*self.posx*coord2pix_x)
        self.py = int(384 - 0.5*self.posy*coord2pix_y)
        return self.px, self.py

    def find_non_overlapping_random_point(self, point_list, w, h):
        done = False
        while not done:
            done = True
            self.find_random_point()
            self.coord2pix()
            p1 = (self.px, self.py)
            for i in range(len(point_list)):
                p2 = point_list[i]
                if self.is_overlap(p1,p2,w,h):
                    done = False
                    break
            if done:
                point_list.append(p1)
        return point_list

    def find_contrasting_colors(self, color_list, color_pallete):
        done = False
        while not done:
            done = True
            color = random.sample(color_pallete,1)[0]
            for i in range(len(color_list)):
                ref_color = color_list[i]
                if np.sqrt(np.sum(np.square(np.subtract(color,ref_color)))) < 250:
                    break
            if done:
                color_list.append(color)

        return color_list

    def spawn_images_on_table(self, num_images = 4, random_table = True, random_target = False):
        if random_table:
            # Randomly choose table texture
            table_texture_path_list = glob.glob("data/table_textures/*.png")
            table_texture_path = random.sample(table_texture_path_list,1)[0]
            print(table_texture_path)
            #table_texture_path = 'data/table_textures/pic_033.png' # - Example of a grayscale table texture image
        else:
            table_texture_path = 'data/table_textures/pic_056.png' # Marble
            #table_texture_path = 'data/table_textures/wpic_019.png' # Wooden Texture
            #table_texture_path = 'data/table_textures/mpic_018.png' # Metal
        table_img = Image.open(table_texture_path)
        #Ensure that the image is in RGB format not L
        table_img = table_img.convert('RGB')
        table_img = table_img.resize((1024,768))
        table_pixels = table_img.load()

        img_folders = glob.glob('data/omniglot_data/training/*/*')
        # Randomly choose the objects and their colors
        #Sample one character more in case target character is present even amongst the distractors
        # Relevant only for the case when target is explicitly defined and not randomly chosen
        img_folder_list = random.sample(img_folders,num_images+1)
        color_pallete = [(c1,c2,c3) for c1 in range(0,255,20) for c2 in range(0,255,20) for c3 in range(0,255,20)]
        char_scale_list = [(s,s) for s in range(100,200,25)]

        if not random_target:
            target_path = 'data/omniglot_data/training/Arcadian/01/0001_01.png'
            target_folder = '/'.join(target_path.split('/')[-3:-1])
            target_color = (0,0,255)
            target_scale = (100,100)
            # Remove character from list if it matches target character
            for i in range(num_images):
                img_folder = '/'.join(img_folder_list[i].split('/')[-2:])
                print(img_folder)
                if img_folder == target_folder:
                    img_folder_list.remove(img_folder_list[i])

        point_list = []
        color_list = self.color_list
        if not len(color_list):
            if target_color:
                color_list = [target_color]
            for i in range(num_images):
                # Find high contrast colors - avoid highly  similar looking colors
                color_list = self.find_contrasting_colors(color_list, color_pallete)

        for i in range(num_images):
            img_list = glob.glob(img_folder_list[i] + '/*.png')
            img_path = random.sample(img_list,1)[0]
            color = color_list[i]

            (w,h) = random.sample(char_scale_list,1)[0]

            # Find location to spawn
            point_list = self.find_non_overlapping_random_point(point_list, 150, 150)

            # Change Target character related properties to something non-random if required
            if i == 0 and not random_target:
                img_path = target_path
                #color = target_color
                (w,h) = target_scale

            if i == 0:
                # Target location to reach for
                self.target_position = [self.posx, self.posy, 0.01]

            alpha_img = Image.open(img_path)
            alpha_img = alpha_img.resize((w,h))
            img = np.array(alpha_img.getdata()).reshape(w, h, 3)
            # Load Image and transform to a 2D numpy array.
            image_array = np.reshape(img, (w * h, 3))

            # Get labels for all points
            kmeans = KMeans(n_clusters=2)
            kmeans.cluster_centers_ = np.asarray([[0, 0, 0],[255,255,255]])
            labels = kmeans.predict(image_array)

            alpha_pixels = alpha_img.load()

            mx, my = point_list[i][0], point_list[i][1]
            lx = mx - int(w/2)
            ly = my - int(h/2)

            #Where ever we have label one replace with table pixels
            label_idx = 0
            for y in range(h):
                for x in range(w):
                    px = lx + x
                    py = ly + y
                    if labels[label_idx] == 0:
                        #print(alpha_pixels[w,h])
                        alpha_pixels[x,y] = color
                    else:
                        if (px < 1024 and px > -1) and (py < 768 and py > -1):
                            alpha_pixels[x,y] = table_pixels[px,py]
                    label_idx += 1

            table_img.paste(alpha_img, (lx,ly))
        #table_img.show()

        # Save the drawn patterns to texture that will be loaded
        table_img.save("table/table.png")

        # Load Table
        self.tableID = self._p.loadURDF("table/table.urdf", basePosition=[0, 0, -0.61], baseOrientation=[0, 0, 0, 1])

    def spawn_colors_on_table(self, num_objects = 4, random_table = False, color_mode = 0, scale_mode = 0, object_type = "cross", \
                            target_color = None, target_scale = None, color_list = None, scale_list = None):
        # color_mode 0 - All random color list
        # color_mode 1 - Distractor random color list with user specified target color
        # color_mode 2 - user input specified color list

        # scale_mode 0 - All random scale list
        # scale_mode 1 - Distractor random scale list with user specified target scale
        # scale_mode 2 - user input specified scale list

        color_list = self.color_list
        scale_list = self.scale_list
        if color_mode == 0:
            random_target_color = True
            random_distractor_color = True
        elif color_mode == 1:
            random_target_color = False
            random_distractor_color = True
            assert target_color is not None
        else:
            random_target_color = False
            random_distractor_color = False
            assert color_list is not None
            assert len(color_list) == num_objects

        if scale_mode == 0:
            random_target_scale = True
            random_distractor_scale = True
        elif scale_mode == 1:
            random_target_scale = False
            random_distractor_scale = True
            assert target_scale is not None
        else:
            random_target_scale = False
            random_distractor_scale = False
            assert scale_list is not None
            assert len(scale_list) == num_objects

        if random_table:
            # Randomly choose table texture
            table_texture_path_list = glob.glob("data/table_textures/*.png")
            table_texture_path = random.sample(table_texture_path_list,1)[0]
            #table_texture_path = 'data/table_textures/pic_033.png' # - Example of a grayscale table texture image
        else:
            table_texture_path = 'data/table_textures/pic_056.png' # Marble
            #table_texture_path = 'data/table_textures/wpic_019.png' # Wooden Texture
            #table_texture_path = 'data/table_textures/mpic_018.png' # Metal
        table_img = Image.open(table_texture_path)
        #Ensure that the image is in RGB format not L
        table_img = table_img.convert('RGB')
        table_img = table_img.resize((1024,768))
        draw = ImageDraw.Draw(table_img)

        if color_mode == 0: # All random colors
            color_list = random.sample(self.color_pallete,num_objects)
        elif color_mode == 1:# All random colors except target color
            # Sample one color more than number of objects as target color might be sampled twice
            color_list = random.sample(self.color_pallete,num_objects+1)
            # Remove target color from colors sampled
            if not random_target_color:
                if target_color in color_list:
                    color_list.remove(target_color)
                color_list[0] = target_color

        if scale_mode == 0:
            scale_list = random.sample(range(10,20,2),num_objects)
        elif scale_mode ==1:
            # Sample one scale more than number of objects as target scale might be sampled twice
            scale_list = random.sample(range(10,20,2),num_objects+1)
            # Remove target color from colors sampled
            if not random_target_scale:
                if target_scale in scale_list:
                    scale_list.remove(target_scale)
                scale_list[0] = target_scale

        object_list = ["circle", "cross", "square", "triangle", "star"]
        if object_type is None:
            object_type = random.sample(object_list,1)[0]

        point_list = []
        for i in range(num_objects):
            # Find location to spawn
            point_list = self.find_non_overlapping_random_point(point_list, 150, 150)
            if i == 0:
                # Target location to reach for
                self.target_position = [self.posx, self.posy, 0.01]

            mx, my = point_list[i][0], point_list[i][1]

            self.DrawObject(draw_obj=draw, object_type=object_type, color=color_list[i], pos = (mx,my), scale = scale_list[i])

        # Save the drawn patterns to texture that will be loaded
        table_img.save("table/table.png")

        # Load Table
        self.tableID = self._p.loadURDF("table/table.urdf", basePosition=[0, 0, -0.61], baseOrientation=[0, 0, 0, 1])

    def reset(self, initial_state = None):
        self.robot_specific_reset(initial_state)
        self._p.stepSimulation()
        self.stateId = self._p.saveState()

        self.state = self.calc_state()
        self.potential = self.calc_potential()
        
        return self.state

    def close(self):
        if self._p is not None:
            self._p.disconnect()
            self._p = None

    def render(self, close = False, mode = "rgb_array"):
        if mode != "rgb_array":
            self.rgb_img = np.array([])
            self.depth_img = np.array([])
        else:
            rgb_img, depth_img = self._p.getCameraImage(self.gif_width, self.gif_height, self.viewMatrix, self.projMatrix, renderer = self._p.ER_BULLET_HARDWARE_OPENGL)[2:4]
            self.rgb_img = np.array(rgb_img).reshape((self.gif_width, self.gif_height, 4))
            self.depth_img = np.array(depth_img).reshape((self.gif_width, self.gif_height))
        return self.rgb_img[:,:,:3]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def addToScene(self, bodies):
        if self.parts is not None:
            parts = self.parts
        else:
            parts = {}

        if self.jdict is not None:
            joints = self.jdict
        else:
            joints = {}

        if self.ordered_joints is not None:
            ordered_joints = self.ordered_joints
        else:
            ordered_joints = []

        dump = 1
        for i in range(len(bodies)):
            print(i)
            if p.getNumJoints(bodies[i]) == 0:
                part_name, robot_name = p.getBodyInfo(bodies[i], 0)
                robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(part_name, bodies, i, -1)
            for j in range(p.getNumJoints(bodies[i])):
                _,joint_name,_,_,_,_,_,_,_,_,_,_,part_name,_,_,_,_ = p.getJointInfo(bodies[i], j)

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                if dump: print("ROBOT PART '%s'" % part_name)
                if dump: print("ROBOT JOINT '%s'" % joint_name) # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )

                parts[part_name] = BodyPart(part_name, bodies, i, j)

                if part_name == self.robot_name:
                    self.robot_body = parts[part_name]

                if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
                    parts[self.robot_name] = BodyPart(self.robot_name, bodies, 0, -1)
                    self.robot_body = parts[self.robot_name]

                if joint_name[:8] != "jointfix":
                    joints[joint_name] = Joint(joint_name, bodies, i, j)
                    ordered_joints.append(joints[joint_name])

                    if joint_name[:6] == "ignore":
                        joints[joint_name].disable_motor()
                        continue

                    joints[joint_name].power_coef = 100.0

        return parts, joints, ordered_joints, self.robot_body

    def calc_potential(self):
        return 0

    def HUD(self, state, a, done):
        pass

class Pose_Helper: # dummy class to comply to original interface
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return p.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()

class BodyPart:
    def __init__(self, body_name, bodies, bodyIndex, bodyPartIndex):
        self.bodies = bodies
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        self.initialPosition = self.current_position()
        self.initialOrientation = self.current_orientation()
        self.bp_pose = Pose_Helper(self)

    def state_fields_of_pose_of(self, body_id, link_id=-1):  # a method you will most probably need a lot to get pose and orientation
        if link_id == -1:
            (x, y, z), (a, b, c, d) = p.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = p.getLinkState(body_id, link_id)
        return np.array([x, y, z, a, b, c, d])

    def get_pose(self):
        return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), _ = p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            (x,y,z), (a,b,c,d), _,_,_,_, (vx, vy, vz), (vr,vp,vy) = p.getLinkState(self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def reset_position(self, position):
        p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, self.get_orientation())

    def reset_orientation(self, orientation):
        p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.get_position(), orientation)

    def reset_pose(self, position, orientation):
        p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], position, orientation)

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        return p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:
    def __init__(self, joint_name, bodies, bodyIndex, jointIndex):
        self.bodies = bodies
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_name
        _,_,_,_,_,_,_,_,self.lowerLimit, self.upperLimit,_,_,_,_,_,_,_ = p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
        self.power_coeff = 0

    def set_state(self, x, vx):
        p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

    def current_position(self): # just some synonyme method
        return self.get_state()

    def current_relative_position(self):
        pos, vel,_ = self.get_state()
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit);
        return (
            2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit),
            0.1 * vel
        )

    def get_state(self): # applied torque is the torque applied to get to the current state
        x, vx,_,applied_torque = p.getJointState(self.bodies[self.bodyIndex],self.jointIndex)
        return x, vx, applied_torque

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _,r = self.get_state()
        return r

    def get_velocity(self):
        _, vx = self.get_state()
        return vx

    def set_position(self, position):
        p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,p.POSITION_CONTROL, targetPosition=position)

    def set_velocity(self, velocity):
        p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,p.VELOCITY_CONTROL, targetVelocity=velocity)

    def set_motor_torque(self, torque): # just some synonyme method
        self.set_torque(torque)

    def set_torque(self, torque):
        p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex], jointIndex=self.jointIndex, controlMode=p.TORQUE_CONTROL, force=torque) #, positionGain=0.1, velocityGain=0.1)

    def reset_current_position(self, position, velocity): # just some synonyme method
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        p.resetJointState(self.bodies[self.bodyIndex],self.jointIndex,targetValue=position, targetVelocity=velocity)
        self.disable_motor()

    def disable_motor(self):
        p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,controlMode=p.VELOCITY_CONTROL, force=0)
