import React from 'react'
import { useNavigate } from 'react-router-dom';
import { Navbar } from './Navbar'
import originalimg from '../public/original.png'
import fakeimg from '../public/fake.png'
import leftarr from '../public/leftarr.svg'
import rightarr from '../public/rightarr.svg'
import './Home.css'

export const Home = () => {
    const navigate = useNavigate(); 
  return (
    <>
    <Navbar />
        <div className='home-container'>
            <div className='home-main'>
                <span>Detect Deepfakes Instantly with</span><br></br>
                <span className='home-main-2'>AI-Powered Precision</span>
            </div>
            <div className='home-sub'>
                <span>Uncover deepfake videos with cutting-edge</span><br></br>
                <span className='home-main-2'>AI detection technology</span>
            </div>
            <div className='home-btn'>
                <button  onClick={() => navigate('/dragdrop')}>Try Demo Now</button>
            </div>
            <div className='home-imgs'>
                <span className='Arrow-left'>
                <span className='Arrow-index-1'>original</span> <img src={leftarr} /></span>
                <img src={originalimg} width="190px" height="230px"/>
                <img src={fakeimg} width="200px" height="230px"/>
                <span className='Arrow-right'>
                 <img src={rightarr} /> <span className='Arrow-index-2'>fake</span></span>
            </div>
        </div>

    </>
  )
}
