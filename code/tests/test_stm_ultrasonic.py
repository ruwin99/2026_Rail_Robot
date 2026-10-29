import unittest
from stm32_host import CORE, run_scenario

class UltrasonicTests(unittest.TestCase):
    def test_edges_hold_resume(self):
        self.assertTrue((CORE/'Src/ultrasonic.c').exists(), 'ultrasonic implementation missing')
        self.assertIn('PASS ultrasonic', run_scenario('ultrasonic'))

    def test_states_and_transition_pause(self):
        self.assertIn('PASS ultrasonic_states', run_scenario('ultrasonic_states'))

    def test_noise_timeout_startup(self):
        self.assertIn('PASS ultrasonic_edges', run_scenario('ultrasonic_edges'))

    def test_prestart_samples_do_not_hold_but_start_is_guarded(self):
        self.assertIn('PASS ultrasonic_prestart', run_scenario('ultrasonic_prestart'))

    def test_pre_mission_near_samples_do_not_leak_into_mission_interlock(self):
        self.assertIn('PASS ultrasonic_mission_gate', run_scenario('ultrasonic_mission_gate'))

    def test_pid_disabled_interlock(self):
        self.assertIn('PASS ultrasonic', run_scenario('ultrasonic', pid_enabled=False))

    def test_gpio_and_interrupt_wiring(self):
        project = CORE.parent
        main = (CORE/'Src/main.c').read_text(encoding='utf-8')
        irq = (CORE/'Src/stm32f1xx_it.c').read_text(encoding='utf-8')
        ioc = (project/'encoder.ioc').read_text(encoding='utf-8')
        trig = main.split('/*Configure GPIO pin : PB8')[1].split('/*Configure GPIO pin : PB9')[0]
        self.assertIn('GPIO_MODE_OUTPUT_PP', trig)
        self.assertIn('GPIO_PIN_RESET', trig)
        self.assertIn('GPIO_MODE_IT_RISING_FALLING', main)
        self.assertIn('HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_9)', irq)
        self.assertIn('Ultrasonic_Tick50us();', irq)
        self.assertIn('PB8.Signal=GPIO_Output', ioc)
        self.assertIn('PB9.GPIO_ModeDefaultEXTI=GPIO_MODE_IT_RISING_FALLING', ioc)
        self.assertIn('NVIC.EXTI9_5_IRQn=true', ioc)

    def test_obstacle_red_active_low(self):
        self.assertIn('PASS ultrasonic', run_scenario('ultrasonic', rgb_active_low=True))
