# import Util.Location
# from overrides import override

import Actor
import Vehicle
import TrafficLight

from Util.Utils import getClassName

NORMALIZATION_RATIO = 2

class Game:
    def __init__(self, delta_t: float):
        self.delta_t = delta_t
        self.simulation_time = 0
        self.actors = []
        self.frame = 0

    """
    it is probably better to update status of traffic lights first
    before updating status of ego_vehicle
    @action is currently not used
    """
    def tick(self, action=None) -> None:
        self.frame += 1
        self._update_traffic_lights()
        if (not action):
            nextLight = self._find_next_light()
            if (nextLight):
                self.actors[self._get_ego_vehicle_index()].tick(nextLight)
        return None

    """
    @action the action agent takes, action: np.ndarray, shape=(1,)
    represents the acceleration of ego_vehicle
    should be a continues value in [-max_deacceleration, max_acceleration]
    """
    def rl_tick(self, action) -> None:
        self.frame += 1
        self._update_traffic_lights()
        if (action):
            i = self._get_ego_vehicle_index()
            self.actors[i].accelerateAt(action[0] * NORMALIZATION_RATIO )
        return None

    def spawn_vehicle(self, vehicle: Actor) -> None:
        self.actors.append(vehicle)
        return None

    def add_traffic_light(self, trafficLight: TrafficLight) -> bool:
        if (getClassName(str(type(trafficLight))) != "TrafficLight"):
            return False
        self.actors.append(trafficLight)
        return True

    def get_delta_t(self) -> float:
        return self.delta_t

    # @staticmethod
    def get_simulation_time(self) -> float:
        return self.simulation_time

    

    def getFrame() -> int:
        return self.frame

    def numTrafficLightAhead(self, actor: Actor) -> int:
        i = self._get_ego_vehicle_index()
        n = 0
        for a in self.actors:
            if (self._find_Actor_Type(a) == "TrafficLight"):
                if (a.getLocation() > self.actors[i].getLocation()):
                    n += 1
        return n

    def reset(self, seed=None) -> None:
        self.frame = 0
        self.simulation_time = 0
        for a in self.actors:
            a.reset()
        # print("world", seed)
        return None




###############################################################################

    """
        private methods
    """

    def _find_next_light(self) -> TrafficLight:
            if (len(self.actors) < 2):
                return None
            vehicle = None
            closest = None
            for actor in self.actors:
                if (getClassName(str(type(actor))) == "Vehicle"):
                    vehicle = actor
                if (getClassName(str(type(actor))) == "TrafficLight"):
                    if (actor.getLocation() > vehicle.getLocation()):
                        if (closest != None):
                            if (actor.getLocation() < closest.getLocation()):
                                closest = actor
                        else:
                            closest = actor
            return closest

    def _find_Actor_Type(self, actor: Actor) -> str:
        return getClassName(str(type(actor)))

    def _get_ego_vehicle_index(self) -> int:
            i = 0
            for actor in self.actors:
                if (getClassName(str(type(actor)))  == "Vehicle"):
                    return i
                i += 1
            # raise error "ego vehicle not found"
            if (not self.actors): print("actors is empty")
            else: print("eg not found")
            return None

    def _update_traffic_lights(self) -> None:
        for light in self.actors:
            if (self._find_Actor_Type(light) == "TrafficLight"):
                light.tick()
        return None